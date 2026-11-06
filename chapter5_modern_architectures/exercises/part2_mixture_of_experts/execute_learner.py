"""Prepare public-reference practice or validate/explicitly execute a completed copy.

Default import and CHECK use stdlib AST/JSON only. Execution never fills answers.
Structural routing checks are not a sandbox or proof of arbitrary Python behavior.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import json
import os
import tempfile
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
CONTRACT_SHA256 = '1cfeca294f383efb5f949fa1bc00ad17218464ed760db1a76cf13aa9b83adce0'
DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
MATH_IMPORTS = frozenset({'math', 'itertools', 'functools', 'collections', 'copy', 'statistics',
                         'typing', 'operator', 'numpy', 'torch', 'torch.nn', 'torch.nn.functional'})
UNSUPPORTED_NAMES = frozenset({'tests', 'reference_solutions',
                               '__import__', 'exec', 'eval', 'globals', 'locals'})


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _digest(value):
    return digest_bytes(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode())


def _source(cell):
    value = cell['source']
    require(type(value) is str or type(value) is list and all(type(v) is str for v in value), 'unsupported cell source')
    return value if type(value) is str else ''.join(value)


def _ast(node):
    return ast.dump(node, include_attributes=False)


def _span(text, node):
    """Top-level definition including decorators and its complete last line."""
    lines = text.encode().splitlines(keepends=True)
    first = min([node.lineno] + [v.lineno for v in getattr(node, 'decorator_list', [])])
    require(node.col_offset == 0, 'only complete top-level answer occurrences are supported')
    # Mixed definition/test statements on the same line are unsupported, rather
    # than silently swallowing a fixed test during substitution.
    tail = lines[node.end_lineno - 1][node.end_col_offset:].strip()
    require(not tail or tail.startswith(b'#'), 'unsupported code after definition on its last line')
    start = sum(map(len, lines[:first - 1]))
    end = sum(map(len, lines[:node.end_lineno]))
    return start, end


def _interface(node):
    """Freeze names, defaults, decorators and class method occurrences; allow annotations."""
    if isinstance(node, ast.ClassDef):
        members = []
        for child in node.body:
            if isinstance(child, ast.Expr) and isinstance(child.value, ast.Constant) and isinstance(child.value.value, str):
                continue
            require(isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)), 'unsupported class-level answer layout')
            members.append(_interface(child))
        return {'kind': 'class', 'name': node.name, 'bases': [_ast(x) for x in node.bases],
                'keywords': [_ast(x) for x in node.keywords],
                'decorators': [_ast(x) for x in node.decorator_list], 'methods': members}
    args = deepcopy(node.args)
    for item in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
        item.annotation = None
        item.type_comment = None
    for item in [args.vararg, args.kwarg]:
        if item is not None:
            item.annotation = None
            item.type_comment = None
    return {'kind': type(node).__name__, 'name': node.name, 'args': _ast(args),
            'decorators': [_ast(x) for x in node.decorator_list],
            'type_params': [_ast(x) for x in getattr(node, 'type_params', [])]}


def _call_interface(value):
    """Compare parameter/default/class shape, preserving separately bound decorators."""
    result = deepcopy(value)
    result.pop('decorators', None)
    if 'methods' in result:
        result['methods'] = [_call_interface(v) for v in result['methods']]
    return result


def _placeholder(node):
    if any(isinstance(v, ast.Raise) and 'NotImplementedError' in ast.unparse(v) for v in ast.walk(node)):
        return True
    for child in ast.walk(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = [v for v in child.body if not (isinstance(v, ast.Expr) and
                    isinstance(v.value, ast.Constant) and isinstance(v.value.value, str))]
            if not body or all(isinstance(v, ast.Pass) or isinstance(v, ast.Expr) and
                              isinstance(v.value, ast.Constant) and v.value.value is Ellipsis for v in body):
                return True
    return False


def _answer_policy(node):
    require(not _placeholder(node), 'remaining implementation stub')
    for child in ast.walk(node):
        require(not isinstance(child, ast.Global), 'unsupported namespace manipulation')
        if isinstance(child, ast.Name):
            require(child.id not in UNSUPPORTED_NAMES, 'unsupported reference/delegation or dynamic namespace: ' + child.id)
        if isinstance(child, ast.Import):
            require(all(v.name in MATH_IMPORTS for v in child.names), 'unreviewed answer import')
        if isinstance(child, ast.ImportFrom):
            require(child.level == 0 and child.module in MATH_IMPORTS and
                    all(v.name != '*' for v in child.names), 'unreviewed answer import')


def build_plan(source):
    """Bind each source occurrence to its unambiguous closest own public answer."""
    require(source.get('nbformat') == 4 and type(source.get('cells')) is list, 'unsupported notebook')
    answers, stubs, occurrences = [], [], []
    for cell_index, cell in enumerate(source['cells']):
        text = _source(cell)
        if cell['cell_type'] == 'markdown':
            for block_index, block in enumerate(re.findall(r'<summary>Solution</summary>(.*?)</details>', text, re.S)):
                for code_index, code in enumerate(re.findall(r'```python\s*\n(.*?)```', block, re.S)):
                    for node in ast.parse(code).body:
                        if isinstance(node, DEFINITIONS):
                            start, end = _span(code, node)
                            answers.append({'cell': cell_index, 'block': block_index, 'code': code_index,
                                            'name': node.name, 'node': node,
                                            'text': code.encode()[start:end].decode(),
                                            'block_sha256': digest_bytes(code.encode())})
        elif cell['cell_type'] == 'code':
            for statement_index, node in enumerate(ast.parse(text).body):
                if isinstance(node, DEFINITIONS) and _placeholder(node):
                    stubs.append({'cell': cell_index, 'statement': statement_index, 'node': node})
    # Assign each public answer to its closest same-name stub occurrence. This
    # supports ARENA's following answers and preceding answers, without borrowing
    # a later exercise's answer when a duplicate-name exercise loses its own.
    assigned = {}
    for answer in answers:
        candidates = [s for s in stubs if s['node'].name == answer['name']]
        if not candidates:
            continue
        distance = min(abs(s['cell'] - answer['cell']) for s in candidates)
        closest = [s for s in candidates if abs(s['cell'] - answer['cell']) == distance]
        require(len(closest) == 1, 'ambiguous visible answer occurrence: ' + answer['name'])
        key = (closest[0]['cell'], closest[0]['statement'])
        assigned.setdefault(key, []).append(answer)
    for stub in stubs:
        cell_index, statement_index, node = stub['cell'], stub['statement'], stub['node']
        candidates = assigned.get((cell_index, statement_index), [])
        require(len(candidates) == 1, 'missing or ambiguous visible answer occurrence: ' + node.name)
        answer = candidates[0]
        require(_call_interface(_interface(node)) == _call_interface(_interface(answer['node'])),
                'public answer changes declared call interface: ' + node.name)
        _answer_policy(answer['node'])
        text = _source(source['cells'][cell_index])
        start, end = _span(text, node)
        occurrences.append({'cell': cell_index, 'statement': statement_index, 'name': node.name,
            'stub_source_sha256': digest_bytes(text.encode()[start:end]), 'stub_ast': _ast(node),
            'interface': _interface(node), 'public_answer_interface': _interface(answer['node']), 'answer_cell': answer['cell'], 'answer_block': answer['block'],
            'answer_code': answer['code'], 'answer_block_sha256': answer['block_sha256'],
            'answer_source': answer['text'], 'answer_source_sha256': digest_bytes(answer['text'].encode())})
    require(occurrences, 'no supported implementation occurrences')
    return {'schema_version': 1, 'source_notebook_object_sha256': _digest(source), 'occurrences': occurrences}


def _bound(source, plan):
    require(plan == build_plan(source), 'source/answer plan changed')


def _replace(text, replacements):
    raw = text.encode()
    for start, end, value in sorted(replacements, reverse=True):
        raw = raw[:start] + value + raw[end:]
    return raw.decode()


def _static_top(book):
    result = {k: deepcopy(v) for k, v in book.items() if k != 'cells'}
    info = result.get('metadata', {}).get('language_info')
    if isinstance(info, dict):
        result['metadata']['language_info'] = {'name': info.get('name')}
    return result


def _static_cell(cell):
    result = deepcopy(cell)
    if result['cell_type'] == 'code':
        result.pop('outputs', None)
        result.pop('execution_count', None)
        for key in ['execution', 'trusted']:
            result.get('metadata', {}).pop(key, None)
    result.pop('source')
    return result


def validate_completed(source, submitted, plan):
    """Validate fixed experiment/source and freeze actual independently written answers."""
    _bound(source, plan)
    require(type(submitted) is dict and set(submitted) == set(source), 'notebook top-level layout changed')
    require(_static_top(submitted) == _static_top(source), 'notebook top metadata changed')
    ids = [c['id'] for c in submitted['cells'] if 'id' in c]
    require(all(type(v) is str and v for v in ids) and len(ids) == len(set(ids)), 'invalid runtime cell IDs')
    require(len(submitted['cells']) == len(source['cells']), 'cell count/order changed')
    grouped = {}
    for occurrence in plan['occurrences']:
        grouped.setdefault(occurrence['cell'], []).append(occurrence)
    actual = []
    for index, (old, new) in enumerate(zip(source['cells'], submitted['cells'], strict=True)):
        old_static, new_static = _static_cell(old), _static_cell(new)
        if 'id' not in old_static:
            new_static.pop('id', None)  # Kernel/notebook serialization may add absent IDs.
        require(old_static == new_static, 'fixed cell identity/metadata changed')
        if index not in grouped:
            require(old['source'] == new['source'], 'nonanswer cell source/order changed')
            continue
        before_text, after_text = _source(old), _source(new)
        before_nodes, after_nodes = ast.parse(before_text).body, ast.parse(after_text).body
        compile(after_text, '<submitted-answer-cell>', 'exec')  # Compile only; no code is executed.
        require(len(before_nodes) == len(after_nodes), 'answer cell statement layout changed')
        before_spans, after_spans = [], []
        for row in grouped[index]:
            original = before_nodes[row['statement']]
            node = after_nodes[row['statement']]
            require(isinstance(node, DEFINITIONS) and
                    _interface(node) in (row['interface'], row['public_answer_interface']), 'answer interface changed')
            _answer_policy(node)
            start, end = _span(before_text, original)
            before_spans.append((start, end, b'__BOUND_ANSWER__\n'))
            start, end = _span(after_text, node)
            after_spans.append((start, end, b'__BOUND_ANSWER__\n'))
            actual.append({'cell': index, 'statement': row['statement'], 'name': row['name'],
                           'source_sha256': digest_bytes(after_text.encode()[start:end]),
                           'source': after_text.encode()[start:end].decode(),
                           'actual_signature_ast': _ast(node.args) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else _ast(node)})
        require(_replace(before_text, before_spans) == _replace(after_text, after_spans),
                'fixed tests or nonanswer experiment code changed')
    return {'status': 'completed_source_validated', 'executed': False, 'authorship': 'not_inferred',
            'answer_occurrences': len(actual), 'answers': actual,
            'submitted_answer_sha256': _digest([{k: v for k, v in row.items() if k != 'actual_signature_ast'} for row in actual]),
            'public_reference_answer_sha256': _digest([{'cell': r['cell'], 'statement': r['statement'],
                'name': r['name'], 'source_sha256': r['answer_source_sha256'], 'source': r['answer_source']}
                for r in plan['occurrences']]),
            'submitted_notebook_object_sha256': _digest(submitted),
            'limitations': 'Structural source/interface checks, not a code sandbox, semantic proof, authorship or mastery assessment.'}


def prepare_public_reference(source, plan):
    """Explicitly fill a distinct exposed-reference practice copy; never execute it."""
    _bound(source, plan)
    result = deepcopy(source)
    grouped = {}
    for row in plan['occurrences']:
        grouped.setdefault(row['cell'], []).append(row)
    for index, rows in grouped.items():
        old = source['cells'][index]
        text = _source(old)
        nodes = ast.parse(text).body
        replacements = []
        for row in rows:
            start, end = _span(text, nodes[row['statement']])
            replacements.append((start, end, (row['answer_source'].rstrip('\n') + '\n').encode()))
        changed = _replace(text, replacements)
        result['cells'][index]['source'] = changed.splitlines(keepends=True) if type(old['source']) is list else changed
    checked = validate_completed(source, result, plan)
    return result, {'status': 'public_reference_source_prepared', 'practice_kind': 'exposed_public_reference_completion',
                    'prior_exposure': 'public_answers_supplied', 'authorship': 'not_inferred', 'executed': False,
                    'answer_occurrences': checked['answer_occurrences'],
                    'prepared_notebook_object_sha256': checked['submitted_notebook_object_sha256']}


def load_frozen():
    contract_path = HERE / 'own_execution_contract.json'
    require(digest_bytes(contract_path.read_bytes()) == CONTRACT_SHA256, 'execution source contract changed')
    contract = json.loads(contract_path.read_text())
    pins = contract['source_files']
    for relative, pin in pins.items():
        path = ROOT / relative
        require(path.resolve().is_relative_to(ROOT) and not path.is_symlink(), 'source path escaped root')
        raw = path.read_bytes()
        require({'sha256': digest_bytes(raw), 'bytes': len(raw)} == pin, 'frozen source file changed: ' + relative)
    source = json.loads((ROOT / contract['exercise_notebook']).read_text())
    _bound(source, contract['plan'])
    return source, contract['plan'], pins


def _save(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(prefix='.own-receipt-', dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write((json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode())
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _run_kernel(book, directory):
    """The only scientific-runtime boundary; reachable solely from explicit --execute."""
    import nbformat
    from nbclient import NotebookClient

    notebook = nbformat.from_dict(deepcopy(book))
    client = NotebookClient(notebook, timeout=3600, kernel_name='python3',
                            resources={'metadata': {'path': str(directory / 'source_snapshot')}})
    try:
        kernel_environment = os.environ.copy()
        workspace = directory / 'source_snapshot'
        kernel_environment['PYTHONPATH'] = os.pathsep.join([str(workspace), str(workspace / 'chapter5_modern_architectures/exercises')])
        client.execute(env=kernel_environment)
    finally:
        nbformat.write(notebook, directory / 'executed_copy.ipynb')
    return json.loads(nbformat.writes(notebook))


def _execute(source, plan, pins, input_path, directory, reference_practice):
    """Never calls public-answer completion; snapshot and validate this supplied copy."""
    require(not directory.exists(), 'output directory exists; preserve it and choose a new directory')
    directory.mkdir(parents=True)
    receipt = {'status': 'VALIDATING', 'authorship': 'not_inferred', 'execution_completed': False,
               'input_kind': 'exposed_reference_practice' if reference_practice else 'submitted_completed_copy',
               'prior_exposure': 'public_answers_supplied' if reference_practice else 'unknown',
               'scientific_acceptance': False, 'automatic_answer_filling': False,
               'source_files': pins, 'execution_workspace': str(directory / 'source_snapshot'),
               'fixed_experiment_uses_canonical_cwd': False, 'kernel_requested': 'python3',
               'cli_python': sys.version, 'entrypoint_sha256': digest_bytes(Path(__file__).read_bytes())}
    try:
        raw = input_path.read_bytes()
        receipt['submitted_input_sha256'] = digest_bytes(raw)
        (directory / 'submitted_input.ipynb').write_bytes(raw)
        submitted = json.loads(raw)
        validated = validate_completed(source, submitted, plan)
        receipt['answer_binding'] = validated
        _save(directory / 'source_notebook.json', source)
        receipt['status'] = 'RUNNING'
        _save(directory / 'execution_receipt.json', receipt)
        require(input_path.read_bytes() == raw, 'submitted copy changed before execution')
        # The unchanged notebook discovers cwd as its root and writes figures there.
        # Use exact copied source paths so it cannot overwrite canonical course assets.
        (directory / 'source_snapshot').mkdir()
        # Every repository dependency is checked again, and copied before runtime starts.
        for relative, pin in pins.items():
            data = (ROOT / relative).read_bytes()
            require(digest_bytes(data) == pin['sha256'], 'source dependency changed before execution')
            snapshot = directory / 'source_snapshot' / relative
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(data)
        result = _run_kernel(submitted, directory)
        require(all(c.get('execution_count') is not None and
                    not any(o.get('output_type') == 'error' for o in c.get('outputs', []))
                    for c in result['cells'] if c['cell_type'] == 'code' and _source(c).strip()),
                'execution did not finish every code cell without errors')
        validate_completed(source, result, plan)
        for relative, pin in pins.items():
            require(digest_bytes((ROOT / relative).read_bytes()) == pin['sha256'], 'source dependency changed during execution')
        receipt.update(status='EXECUTED', execution_completed=True,
                       executed_notebook_object_sha256=_digest(result),
                       kernel_reported_language_info=result.get('metadata', {}).get('language_info', {}))
        # This records supplied completed-source execution, not inferred authorship.
        receipt['completed_from_supplied_answers'] = True
    except BaseException as error:
        receipt.update(status='FAILED', error=repr(error), execution_completed=False)
    _save(directory / 'execution_receipt.json', receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--check', action='store_true')
    action.add_argument('--prepare-reference', action='store_true')
    action.add_argument('--validate-completed', type=Path)
    action.add_argument('--execute', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--reference-practice', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.output_dir is not None:
            # Resolve before creation: kernel cwd and import prefixes must agree.
            args.output_dir = args.output_dir.resolve()
        require(not args.reference_practice or args.execute is not None, '--reference-practice is an explicit execution input label')
        source, plan, pins = load_frozen()
        if args.execute is not None:
            require(args.output_dir is not None, '--execute requires a new --output-dir')
            receipt = _execute(source, plan, pins, args.execute, args.output_dir, args.reference_practice)
        elif args.prepare_reference:
            require(args.output_dir is not None and not args.output_dir.exists(), 'reference preparation requires a new output directory')
            completed, receipt = prepare_public_reference(source, plan)
            args.output_dir.mkdir(parents=True)
            path = args.output_dir / 'public_reference_practice.ipynb'
            _save(path, completed)
            receipt['prepared_file_sha256'] = digest_bytes(path.read_bytes())
            _save(args.output_dir / 'preparation_receipt.json', receipt)
        elif args.validate_completed is not None:
            require(args.output_dir is None, 'validation is read-only; omit --output-dir')
            receipt = validate_completed(source, json.loads(args.validate_completed.read_text()), plan)
        else:
            require(args.output_dir is None, 'CHECK is read-only; omit --output-dir')
            receipt = {'status': 'source_check_passed', 'executed': False, 'authorship': 'not_inferred',
                       'answer_occurrences': len(plan['occurrences']), 'source_files': pins}
        print(json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False))
        return 1 if receipt['status'] == 'FAILED' else 0
    except (ValueError, OSError, SyntaxError, KeyError, TypeError) as error:
        print(json.dumps({'status': 'FAILED', 'executed': False, 'authorship': 'not_inferred', 'error': str(error)}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
