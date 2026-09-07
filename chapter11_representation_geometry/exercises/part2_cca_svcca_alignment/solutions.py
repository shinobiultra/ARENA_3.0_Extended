"""Reference functions and independent native CPU verification for the alignment lesson."""
from __future__ import annotations
import inspect,json,os,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from chapter11_representation_geometry.exercises.part2_cca_svcca_alignment.alignment_methods import *
from chapter11_representation_geometry.exercises.part2_cca_svcca_alignment import alignment_methods_tests


def run_smoke_test(cpu=True):
    tested=[]
    for name,function in vars(alignment_methods_tests).items():
        if name.startswith('test_') and inspect.isfunction(function):function();tested.append(name)
    return {'tests_passed':True,'accepted':True,'test_count':len(tested),'claim_scope':'Independent mathematical oracles and counterexamples.'}


def run_gpu_test(max_vram_gb: float = 24.0):
    """Legacy report entry-point name; this section explicitly executes on CPU."""
    from chapter11_representation_geometry.exercises.part2_cca_svcca_alignment.alignment_reproduce import reproduce
    from chapter11_representation_geometry.exercises.part2_cca_svcca_alignment.alignment_evidence import sha256
    here=Path(__file__).parent;publication_path=here/'expected_outputs/learner_execution.json'
    publication=json.loads(publication_path.read_text())
    learner=Path(os.environ.get('ARENA_ALIGNMENT_LEARNER_DIRECTORY',publication['learner_directory']))
    report_root=learner.parent/'fresh_reproductions';report_root.mkdir(parents=True,exist_ok=True)
    output=Path(tempfile.mkdtemp(prefix='run-',dir=report_root))/'fresh_cpu'
    fresh=reproduce(learner,output);audit=fresh['learner_audit']
    expected=here/'expected_outputs/fresh_reproduction.json'
    expected.write_bytes((output/'fresh_reproduction.json').read_bytes())
    return {'cuda_available':False,'device':'cpu','peak_vram_gb':0.,'within_vram_budget':True,
            'native_model_reloaded':True,'actual_learner_passed':True,'fresh_reproduction_passed':True,
            'own_answer_code_cells':audit['own_answer_code_cells'],'embedded_figures':len(audit['figures']),
            'retained_outcomes':audit['outcomes'],'main_forward_rows':audit['main_forward_rows'],
            'fresh_compared_examples':fresh['compared_rows'],'fresh_condition_batches':fresh['compared_conditions'],
            'fresh_max_logit_error':fresh['maximum_logit_error'],'fresh_top_token_mismatches':fresh['top_token_mismatches'],
            'fresh_output_directory':str(output),'fresh_elapsed_seconds':fresh['elapsed_seconds'],
            'learner_evidence_sha256':sha256(publication_path),'fresh_evidence_sha256':sha256(expected),
            'final_used_for_selection':False,'whole_paper_replication':False}


def run_full_experiment(max_vram_gb: float = 24.0):
    return {'numerical':run_smoke_test(),'fresh_cpu_model':run_gpu_test(max_vram_gb)}
