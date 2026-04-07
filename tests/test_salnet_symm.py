"""Smoke test for scripts/symm_net/salnet_symm.py.

No dataset needed. Runs in tmp_path so pytest cleans up the runs/ output
directory automatically.
"""

import json
import subprocess
import sys

import pytest
from conftest import SCRIPTS_DIR

SCRIPT = SCRIPTS_DIR / "symm_net" / "salnet_symm.py"


@pytest.mark.timeout(60)
def test_salnet_symm(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--n_epochs",
            "1",
            "--len_epoch",
            "100",
            "--output-dir",
            str(tmp_path),
            "--tags",
            "smoketest",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        f"salnet_symm.py exited with code {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )

    metrics_files = list(tmp_path.rglob("metrics.json"))
    assert len(metrics_files) == 1, "metrics.json not found in runs/"
    metrics = json.loads(metrics_files[0].read_text())
    assert "scalars" in metrics
