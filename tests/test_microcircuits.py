"""Smoke test for scripts/microcircuits/run.py.

Uses tests/fixtures/microcircuits_smoke.yaml with reduced number of pattern and
epoch duration, to keep runtime well under the 60 s timeout.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import SCRIPTS_DIR

SCRIPT = SCRIPTS_DIR / "microcircuits" / "run.py"
EXAMPLE_YAML = Path(__file__).parent / "fixtures" / "microcircuits_smoke.yaml"


@pytest.mark.timeout(60)
def test_microcircuits_run(tmp_path):
    config = tmp_path / "example.yaml"
    shutil.copy(EXAMPLE_YAML, config)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(config), "0"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        f"run.py exited with code {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert (tmp_path / "teacher.0000.png").exists(), "teacher plot not created"
    assert (tmp_path / "student.0000.pickle").exists(), "student pickle not created"
