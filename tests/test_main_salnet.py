"""Smoke tests for scripts/symm_net/main_salnet.py.

Must run from scripts/symm_net/ so that `from load_utils import ...` resolves
and `../datasets` points to the cached dataset directory (scripts/datasets/).

Results are written to --output-dir, which the tests point at tmp_path so
pytest handles cleanup automatically.

All tests here are marked pytorch and skipped by default. Enable with --pytorch:
    pytest tests/ --pytorch                    # bp only (default section)
    pytest tests/ --pytorch --sections bp,fa   # specific subset
    pytest tests/ --pytorch --sections all     # all sections

NOTE: first run requires MNIST to be downloaded (~10 MB). In CI, scripts/datasets/
is restored from cache so no download is needed after the first run.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import SCRIPTS_DIR

SYMM_NET_DIR = SCRIPTS_DIR / "symm_net"
SCRIPT = SYMM_NET_DIR / "main_salnet.py"
EXP_SETTINGS = Path(__file__).parent / "fixtures" / "symmnet_smoke.yaml"

ALL_SECTIONS = ["bp", "fa", "bp_w_fa", "akrout", "scfa", "sal", "rdd"]
TIMEOUT = 300


def pytest_generate_tests(metafunc):
    if "section" in metafunc.fixturenames:
        opt = metafunc.config.getoption("--sections")
        sections = ALL_SECTIONS if opt == "all" else [s.strip() for s in opt.split(",")]
        params = [
            pytest.param(s, marks=[pytest.mark.pytorch, pytest.mark.timeout(TIMEOUT)])
            for s in sections
        ]
        metafunc.parametrize("section", params)


def test_main_salnet(tmp_path, section):
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "-f",
            str(EXP_SETTINGS),
            "-s",
            section,
            "--dataset",
            "mnist",
            "--n_epochs",
            "1",
            "--output-dir",
            str(tmp_path),
            "--tags",
            "smoke",
        ],
        capture_output=True,
        text=True,
        cwd=str(SYMM_NET_DIR),
    )

    assert result.returncode == 0, (
        f"main_salnet.py (section={section}) exited with code {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )

    metrics_files = list(tmp_path.rglob("metrics.json"))
    assert len(metrics_files) == 1, "metrics.json not found"
    metrics = json.loads(metrics_files[0].read_text())
    assert "scalars" in metrics
