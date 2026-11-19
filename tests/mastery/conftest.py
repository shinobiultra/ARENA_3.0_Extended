"""Make the repository root importable for the study-hub tests."""
import os
from pathlib import Path
import sys

CANDIDATE_ROOT = Path(os.environ.get('SAE_CANDIDATE_ROOT', str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(CANDIDATE_ROOT))
