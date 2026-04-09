"""Smoke test for scripts/ssn/train_bm.py.

Uses tests/fixtures/ssn_smoke.yaml: 11 epochs, dur=100 (vs 1000 in the
minimal_working_example), to keep runtime well under the 60 s timeout.
Note: first run includes numba JIT compilation (~10-15 s overhead).
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import SCRIPTS_DIR

SCRIPT = SCRIPTS_DIR / "ssn" / "train_bm.py"
FIXTURE_YAML = Path(__file__).parent / "fixtures" / "ssn_smoke.yaml"


@pytest.mark.timeout(60)
def test_train_bm(tmp_path):
    config = tmp_path / "ssn_smoke.yaml"
    shutil.copy(FIXTURE_YAML, config)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(config)],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        f"train_bm.py exited with code {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert (tmp_path / "ssn_smoke.png").exists(), "result figure not created"
    assert (tmp_path / "ssn_smoke.npz").exists(), "result npz not created"
