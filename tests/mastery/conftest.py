"""Scope the root-bound test session anchor to this companion's engineering files."""
import os
from pathlib import Path
import sys
import pytest

CANDIDATE_ROOT=Path(os.environ.get('SAE_CANDIDATE_ROOT',str(Path(__file__).resolve().parents[2])))
sys.path.insert(0,str(CANDIDATE_ROOT))

@pytest.fixture(autouse=True)
def companion_session_anchor(request,tmp_path,monkeypatch):
    if request.node.path.name not in ('test_sae_prediction.py','test_sae_prediction_v2.py'):
        yield;return
    from mastery_hub.units.sae_prediction import records
    assert records.ROOT==CANDIDATE_ROOT, 'regression must import the explicitly selected candidate source'
    # This is the explicit root-bound anchor configuration for DEV tests only.
    # Changing XDG_DATA_HOME within a test cannot alter this authoritative path.
    monkeypatch.setattr(records,'SESSION_ANCHOR',tmp_path/'session-anchor',raising=False)
    monkeypatch.setenv('XDG_DATA_HOME',str(tmp_path/'private'))
    yield
