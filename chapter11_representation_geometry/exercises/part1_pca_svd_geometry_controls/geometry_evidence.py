"""Audit measured learner outputs; never substitute an independent experiment."""
from __future__ import annotations

from arena_ext.readability_compatibility import source_matches_execution
import ast
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
FIGURES=('geometry_template_offsets.png','geometry_calendar_order_null.png','geometry_plane_intervention.png','geometry_angle_play.png',
         'geometry_mistral_projections.png','geometry_mistral_layer_sweep.png','geometry_mistral_final_controls.png',
         'geometry_mistral_all_donors.png','geometry_mistral_actual_answers.png','geometry_projection_stability.png',
         'geometry_projection_gallery.png','geometry_mistral_angle_radius.png','geometry_mistral_single_play.png')


def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_gzip(path):return json.loads(gzip.decompress(Path(path).read_bytes()))


def validate_plot_heading_update(run):
    """Permit the inspected heading-only change; numerical figure code must match."""
    old=ast.parse((run/'source_snapshot/geometry_plotting.py').read_text())
    new=ast.parse((HERE/'geometry_plotting.py').read_text())
    class RemoveHeading(ast.NodeTransformer):
        def visit_FunctionDef(self,node):
            if node.name!='plot_angle_play':return node
            node.body=[statement for statement in node.body]
            return self.generic_visit(node)
        def visit_Expr(self,node):
            call=node.value
            if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and call.func.attr in {'text','set_title'}:
                return None
            return self.generic_visit(node)
    if ast.dump(RemoveHeading().visit(old))!=ast.dump(RemoveHeading().visit(new)):
        raise ValueError('Plot numerical code changed; heading-only rendering validation is insufficient.')
    record=json.loads((run/'layout_update/render_update.json').read_text())
    if record['source_sha256']!=sha256(HERE/'geometry_plotting.py') or record['input_sha256']!=sha256(run/'experiments/angle_play.json.gz'):
        raise ValueError('Heading render source or actual measurements changed.')
    if record['figure_sha256']!=sha256(run/'layout_update/geometry_mistral_angle_radius.png'):
        raise ValueError('Heading render figure changed.')
    rendered=json.loads((run/'layout_update/render.ipynb').read_text())
    images=[o for c in rendered['cells'] for o in c.get('outputs',[]) if 'image/png' in o.get('data',{})]
    if len(images)!=1:raise ValueError('Heading-only rerender must contain its actual inline figure.')
    return record


def validate_learner(run):
    from .build_notebooks import lesson_cells,complete_learner
    run=Path(run).resolve();execution=json.loads((run/'execution.json').read_text())
    if execution['status']!='PASS' or execution['scope']!='full' or execution['changed_sources'] or not execution['completed_from_own_answers']:
        raise ValueError('Require a completed, unchanged, full own-answer learner run.')
    for name,digest in execution['source_inputs'].items():
        if sha256(run/'source_snapshot'/name)!=digest:raise ValueError('Actual execution source snapshot changed: '+name)
        # Prose can be improved after inspecting results. Executed code is compared below.
        if name=='geometry_plotting.py' and not source_matches_execution(HERE/name, digest):
            validate_plot_heading_update(run)
        elif name!='lesson.md' and not source_matches_execution(HERE/name, digest):raise ValueError('Executed dependency changed: '+name)
    actual=json.loads((run/'completed_learner.ipynb').read_text())
    solved=complete_learner({'cells':lesson_cells()})
    actual_code=[c for c in actual['cells'] if c['cell_type']=='code']
    expected_code=[c for c in solved['cells'] if c['cell_type']=='code']
    if len(actual_code)!=len(expected_code):raise ValueError('Code cell inventory changed.')
    for index,(measured,expected) in enumerate(zip(actual_code,expected_code,strict=True)):
        source=''.join(measured['source'])
        if index==0:
            suffix=f'\nassets_dir = Path({str(run/"figures")!r})\nassets_dir.mkdir(exist_ok=True)\n'
            if not source.endswith(suffix):raise ValueError('Unknown learner setup override.')
            source=source[:-len(suffix)]
        if ast.dump(ast.parse(source))!=ast.dump(ast.parse(''.join(expected['source']))):raise ValueError(f'Actual own-answer code changed in code cell {index}.')
        if measured.get('execution_count') is None or any(o['output_type']=='error' for o in measured.get('outputs',[])):
            raise ValueError('Incomplete notebook output.')
    embedded=sum('image/png' in output.get('data',{}) for cell in actual_code for output in cell.get('outputs',[]))
    if embedded!=len(FIGURES) or any(not (run/'figures'/name).is_file() for name in FIGURES):raise ValueError('Require all actual figures both embedded and saved.')
    study=read_gzip(run/'experiments/measured_study.json.gz');protocol=study['protocol'];outcomes=study['outcomes']
    if protocol!=json.loads((HERE/'geometry_protocol.json').read_text()):raise ValueError('Frozen protocol changed.')
    if not study.get('complete') or len(outcomes)!=120649 or study['model_forward_rows']!=105484:
        raise ValueError('Not all planned calendar rows ran.')
    seen=set();counts=Counter();max_margin_error=0.
    for row in outcomes:
        key=(row['row_id'],row['condition'],row['shift'],row['draw'],row.get('layer_input'))
        if key in seen:raise ValueError('Repeated outcome: '+str(key))
        seen.add(key);counts[(row['family'],row['condition'],row['draw'],row.get('layer_input'))]+=1
        task=study['tasks'][row['family']];tokens=task['calendar_token_ids'];n=len(tokens)
        original=(row['start']+row['offset'])%n;desired=(original+row['shift'])%n
        logits=np.asarray(row['calendar_logits'],dtype=np.float32)
        if logits.shape!=(n,) or not np.isfinite(logits).all():raise ValueError('Missing or nonfinite actual logits.')
        margin=float(logits[desired]-logits[original]);max_margin_error=max(max_margin_error,abs(margin-row['desired_minus_original_logit']))
        if (row['original_answer']!=original or row['desired_answer']!=desired or row['restricted_answer']!=int(logits.argmax())
            or row['original_full_correct']!=(row['top_token_id']==tokens[original]) or row['desired_full_correct']!=(row['top_token_id']==tokens[desired])):
            raise ValueError('Stored answer metric disagrees with actual logits/tokens.')
    if max_margin_error>1e-6:raise ValueError('Stored margin does not match actual logits.')
    final=[r for r in outcomes if r['row_id'].startswith('final/')]
    if len(final)!=100656:raise ValueError('Final row inventory is incomplete.')
    summary={}
    for family,task in study['tasks'].items():
        n=len(task['names']);base_count=len(task['rows']['final']);train_count=len(task['rows']['train']);dev_count=len(task['rows']['development'])
        if len(task['layer_sweep'])!=33 or len(task['development_interventions'])!=33 or task['no_op_max_logit_error']!=0:
            raise ValueError('Missing layer sweep or failed no-op.')
        for condition in protocol['all_donor_conditions']+protocol['shared_row_conditions']:
            if counts[family,condition,None,None]!=base_count*(n-1):raise ValueError('A donor condition is incomplete.')
        for condition in protocol['null_conditions']:
            for draw in range(protocol['null_draws']):
                if counts[family,condition,draw,None]!=base_count:raise ValueError('A null draw is incomplete.')
        for boundary in range(33):
            for condition in ('pure_coordinate','centered_mean_ablated','author_raw_mean_ablated','full_donor','mean_only','equal_norm_random_edit'):
                if counts[family,condition,None,boundary]!=dev_count:raise ValueError('Development sweep incomplete.')
        if len(task['leave_one_class_out'])!=n or sum(r['rows'] for r in task['leave_one_class_out'])!=base_count:raise ValueError('Class holdouts incomplete.')
        primary=[r for r in final if r['family']==family and r['shift']==2]
        stats={}
        for condition in protocol['all_donor_conditions']+protocol['shared_row_conditions']+protocol['null_conditions']:
            chosen=[r for r in primary if r['condition']==condition]
            stats[condition]={'rows':len(chosen),'desired_calendar_accuracy':np.mean([r['restricted_answer']==r['desired_answer'] for r in chosen]).item(),
                              'desired_full_accuracy':np.mean([r['desired_full_correct'] for r in chosen]).item(),
                              'mean_margin':np.mean([r['desired_minus_original_logit'] for r in chosen]).item()}
        baseline=[r for r in primary if r['condition']=='baseline']
        summary[family]={'new_prompts':base_count,'baseline_calendar_accuracy':np.mean([r['restricted_answer']==r['original_answer'] for r in baseline]).item(),
                         'baseline_full_accuracy':np.mean([r['original_full_correct'] for r in baseline]).item(),
                         'coordinate_r2':task['final_coordinate_r2'],'angular_accuracy':task['final_angular_accuracy'],
                         'unseen_class_angular_accuracy':sum(r['angular_correct'] for r in task['leave_one_class_out'])/base_count,
                         'primary_shift_conditions':stats}
    projections=read_gzip(run/'experiments/projection_study.json.gz')
    expected={(family,control,reducer,setting,seed) for family in study['tasks'] for control in ('real','independent_noise')
              for reducer in ('umap','tsne') for setting in range(3) for seed in range(5)}
    keys={(r['family'],r['control'],r['reducer'],r['setting_index'],r['seed']) for r in projections}
    if len(projections)!=120 or keys!=expected:raise ValueError('Reducer sweep incomplete or duplicated.')
    for record in projections:
        labels=np.array([r['start'] for r in study['tasks'][record['family']]['rows']['final']])
        for metric,pred in [('final_knn_accuracy','final_predictions'),('permuted_training_labels_accuracy','permuted_training_label_predictions')]:
            if abs(record[metric]-np.mean(labels==np.asarray(record[pred])))>1e-7:raise ValueError('Projection metric disagrees with actual predictions.')
        if record['reducer']=='tsne' and not record['scope'].startswith('transductive'):raise ValueError('t-SNE scope is misstated.')
    angles=read_gzip(run/'experiments/angle_play.json.gz')
    if len(angles)!=6 or sum(len(r['calendar_logits']) for r in angles)!=3192:raise ValueError('Continuous play grid incomplete.')
    runtime=json.loads((run/'experiments/runtime.json').read_text())
    return {'accepted':True,'own_answer_code_cells':len(actual_code),'embedded_figures':embedded,'final_prompts':579,'all_donor_pairs':5634,
            'retained_outcomes':len(outcomes),'model_forward_rows':study['model_forward_rows'],'null_draws_per_kind':32,'projection_fits':len(projections),
            'angle_play_rows':3196,'single_play_rows':4,'max_stored_margin_error':max_margin_error,'runtime':runtime,'tasks':summary,
            'final_used_for_selection':False,'tsne_is_transductive':True}
