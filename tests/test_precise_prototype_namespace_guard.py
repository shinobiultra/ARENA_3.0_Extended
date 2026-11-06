"""Namespace-ban precision and escaped literals; engineering strings only."""
from pathlib import Path
import importlib.util
import pytest
C=Path(__file__).resolve().parents[1]
@pytest.fixture
def guard():
 # Extract only the pure guard definition; never import or execute lesson code.
 import ast, re
 tree=ast.parse((C/'tests/test_extension_sections.py').read_text())
 node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_uses_arena_ext_namespace')
 namespace={'re':re}
 exec(compile(ast.Module(body=[node],type_ignores=[]),'<DEV namespace guard>','exec'),namespace)
 return namespace['_uses_arena_ext_namespace']
@pytest.mark.parametrize('code',[
 'import arena_ext',
 'import arena_ext as hidden',
 'from arena_ext import method',
 'from arena_ext.hidden import method as alias',
 '__import__("arena_ext")',
 'importlib.import_module("arena_ext.hidden")',
 r'__import__("\x61rena_ext")',
 r'importlib.import_module("\u0061rena_ext.hidden")',
 '__import__("arena_" "ext")',
 'module_name = "arena_ext"\nimportlib.import_module(module_name)',
])
def test_actual_namespace_import_alias_dynamic_and_escaped_literals_blocked(guard,code):
 assert guard(code)
@pytest.mark.parametrize('code',[
 'cache = root / ".cache" / "arena_extended" / "treeshap"',
 'arena_extended_cache = {}',
 'cache_arena_extended = "/tmp/arena_extended/data"',
 'from pathlib import Path\ncache = Path("arena_extended")',
])
def test_legacy_cache_component_and_variables_are_not_prototype_dependencies(guard,code):
 assert not guard(code)
def test_old_false_positive_is_recorded_without_editing_input_literal(guard):
 code='cache = root / ".cache" / "arena_extended" / "treeshap"'
 assert 'arena_ext' in code
 assert not guard(code)

@pytest.mark.parametrize('code',[
 "__import__('arena_' + 'ext')",
 "importlib.import_module('arena_' + 'ext.fake')",
 "__import__(('ar' + 'ena_') + ('e' + 'xt'))",
 r"__import__('\x61rena_' + 'ext')",
 "module_name = 'arena' + '_' + 'ext.hidden'",
])
def test_statically_constant_string_additions_blocked(guard, code):
 assert guard(code)

@pytest.mark.parametrize('code',[
 "cache = 'arena_' + 'extended'",
 "module_name = prefix + suffix",
 "module_name = ''.join(['arena_', 'ext'])",
 "__import__('arena_{}'.format(suffix))",
])
def test_bounded_fold_preserves_cache_and_does_not_claim_runtime_dataflow(guard, code):
 assert not guard(code)
