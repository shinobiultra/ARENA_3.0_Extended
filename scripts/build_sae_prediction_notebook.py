#!/usr/bin/env python3
"""Generate the single additive blank SAE companion; --check is read-only."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUTPUT=ROOT/'mastery_hub/units/sae_prediction/exercises.ipynb'
BASELINE='source_baseline_sha256'

def fingerprint(notebook):
    value=deepcopy(notebook)
    value.get('metadata',{}).get('sae_prediction',{}).pop(BASELINE,None)
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def make_notebook():
    from mastery_hub.units.sae_prediction import lesson
    notebook={'nbformat':4,'nbformat_minor':5,'cells':lesson.build_cells(),
              'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                          'language_info':{'name':'python'},'sae_prediction':{'status':'unexecuted_implementation_candidate','automatic_experiment_execution':False}}}
    notebook['metadata']['sae_prediction'][BASELINE]=fingerprint(notebook)
    return notebook

def has_work(raw):
    return ('widgets' in raw.get('metadata',{}) or
      any(('execution' in cell.get('metadata',{}) or 'trusted' in cell.get('metadata',{}) or
           cell.get('outputs') or cell.get('execution_count') is not None)
          for cell in raw.get('cells',[])))

def synchronize(path,expected,*,check):
    path=Path(path)
    if path.is_symlink():return False
    if path.exists():
        try:
            from mastery_hub.units.sae_prediction.records import parse
            old=parse(path.read_text())
            if type(old) is not dict or type(old.get('cells')) is not list or type(old.get('metadata')) is not dict:return False
            if not all(type(c) is dict and type(c.get('metadata',{})) is dict for c in old['cells']):return False
        except (ValueError,UnicodeError):return False
        # No-op always preserves exact bytes, including learner/runtime metadata.
        # JSON representation preserves bool/int/float distinctions; Python == does not.
        if json.dumps(old,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)==json.dumps(expected,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False):return True
        if check or has_work(old):return False
        baseline=old.get('metadata',{}).get('sae_prediction',{}).get(BASELINE)
        if baseline!=fingerprint(old):return False
    elif check:return False
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(expected,indent=1,ensure_ascii=False)+'\n');return True

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true');args=parser.parse_args(argv)
    ok=synchronize(OUTPUT,make_notebook(),check=args.check)
    print(('OK' if ok else 'DRIFT / learner work preserved')+': mastery_hub/units/sae_prediction/exercises.ipynb')
    return 0 if ok else 1
if __name__=='__main__':raise SystemExit(main())
