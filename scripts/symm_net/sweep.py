#!/usr/bin/env python3
"""Sweep launcher for convenient reproduction of data for SymmNet paper figure

Runs dataset × algo × seed combinations.

Sequential run (default):
    python sweep.py

Parallel run with 4 workers:
    python sweep.py --n-workers 4

Subset example:
    python sweep.py --datasets cifar10 --algos bp sal --n-seeds 2

Finished runs (metrics.json present) are skipped, if sweep is executed again.
"""

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ALL_DATASETS = ["cifar10", "fmnist", "svhn"]
ALL_ALGOS = ["bp", "fa", "bp_w_fa", "akrout", "scfa", "sal", "rdd"]
PARAM_FILE = "exp_settings.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=ALL_DATASETS,
        choices=ALL_DATASETS,
        metavar="DS",
        help="Datasets to run (default: all three).",
    )
    p.add_argument(
        "--algos",
        nargs="+",
        default=ALL_ALGOS,
        choices=ALL_ALGOS,
        metavar="ALGO",
        help="Algorithm sections from exp_settings.yaml (default: all seven).",
    )
    p.add_argument(
        "--n-seeds",
        type=int,
        default=5,
        dest="n_seeds",
        help="Number of seeds, numbered 0 … N-1 (default: 5).",
    )
    p.add_argument(
        "--n-workers",
        type=int,
        default=1,
        dest="n_workers",
        help="Number of parallel workers (default: 1 = sequential).",
    )
    p.add_argument(
        "--sweep-name",
        default="sweep",
        dest="sweep_name",
        help="Subdirectory name under --base-dir (default: sweep).",
    )
    p.add_argument(
        "--base-dir",
        default="../../results/symm_net",
        dest="base_dir",
        help="Root output directory (default: ../../results/symm_net).",
    )
    return p.parse_args()


def _run_one(
    run_dir: Path,
    dataset: str,
    algo: str,
    seed: int,
    param_file: str,
) -> tuple[str, int | None]:
    """Run a single training job.

    Returns:
        A (label, returncode) tuple. returncode is None if the run was skipped.
    """
    label = f"{dataset}/{algo}/seed_{seed}"
    if (run_dir / "metrics.json").exists():
        return label, None  # already done
    result = subprocess.run(
        [
            sys.executable,
            "main_salnet.py",
            "-f",
            param_file,
            "-s",
            algo,
            "--dataset",
            dataset,
            "--seed",
            str(seed),
            "--run-dir",
            str(run_dir),
        ],
        check=False,
    )
    return label, result.returncode


def main() -> None:
    args = parse_args()
    sweep_dir = Path(args.base_dir) / args.sweep_name
    total = len(args.datasets) * len(args.algos) * args.n_seeds

    print(f"Sweep: {total} runs → {sweep_dir}")
    print(f"  datasets: {args.datasets}")
    print(f"  algos: {args.algos}")
    print(f"  seeds: 0 … {args.n_seeds - 1}")
    print(f"  workers: {args.n_workers}")
    print()

    runs = [
        (sweep_dir / dataset / algo / f"seed_{seed}", dataset, algo, seed)
        for dataset in args.datasets
        for algo in args.algos
        for seed in range(args.n_seeds)
    ]

    done = 0
    with ThreadPoolExecutor(max_workers=args.n_workers) as pool:
        futures = {
            pool.submit(_run_one, run_dir, dataset, algo, seed, PARAM_FILE): i
            for i, (run_dir, dataset, algo, seed) in enumerate(runs)
        }
        for future in as_completed(futures):
            label, code = future.result()
            done += 1
            if code is None:
                print(f"[{done}/{total}] Skip  {label}")
            elif code != 0:
                print(f"[{done}/{total}] FAIL  {label} (exit {code})")
            else:
                print(f"[{done}/{total}] Done  {label}")

    print(f"\nSweep complete. Results in {sweep_dir}")


if __name__ == "__main__":
    main()
