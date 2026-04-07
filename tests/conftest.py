from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"


def pytest_addoption(parser):
    parser.addoption(
        "--pytorch",
        action="store_true",
        default=False,
        help="Run tests that require PyTorch CNN training (slow on CPU, skipped in CI).",
    )
    parser.addoption(
        "--sections",
        type=str,
        default="bp",
        help=(
            "Comma-separated list of main_salnet sections to test "
            "(e.g. --sections bp,fa,sal). Use 'all' for every section."
        ),
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--pytorch"):
        skip = pytest.mark.skip(
            reason="PyTorch training test — run with --pytorch to include"
        )
        for item in items:
            if "pytorch" in item.keywords:
                item.add_marker(skip)
