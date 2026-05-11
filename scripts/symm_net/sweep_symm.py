#!/usr/bin/env python3
"""Sweep launcher for pure SAL symmetrization experiments (salnet_symm.py).

Iterates over learning rates × seeds. Each combination is stored in a structured
directory tree that plot_puresymm.ipynb can read directly.

Sequential run (default):
    python sweep_symm.py

Parallel run with 4 workers:
    python sweep_symm.py --n-workers 4

Subset example:
    python sweep_symm.py --lrs 0.001 0.01 --n-seeds 2 --n-epochs 10

Finished runs (metrics.json present) are skipped, if sweep is executed again.
"""

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DEFAULT_LRS = [0.01, 0.02, 0.04, 0.08]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--lrs",
        nargs="+",
        type=float,
        default=DEFAULT_LRS,
        metavar="LR",
        help="Learning rates to sweep (default: 0.01 0.02, 0.04, 0.08).",
    )
    p.add_argument(
        "--n-seeds",
        type=int,
        default=5,
        dest="n_seeds",
        help="Number of seeds per learning rate, numbered 0 … N-1 (default: 5).",
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
        default="puresymm",
        dest="sweep_name",
        help="Subdirectory name under --base-dir (default: puresymm).",
    )
    p.add_argument(
        "--base-dir",
        default="../../results/symm_net/puresymm",
        dest="base_dir",
        help="Root output directory (default: ../../results/symm_net/puresymm).",
    )
    # forwarded to salnet_symm.py
    p.add_argument("--n-epochs", type=int, default=2000, dest="n_epochs")
    p.add_argument("--len-epoch", type=int, default=100, dest="len_epoch")
    p.add_argument("--batchsize", type=int, default=64)
    return p.parse_args()


def _run_one(
    run_dir: Path,
    lr: float,
    seed: int,
    n_epochs: int,
    len_epoch: int,
    batchsize: int,
) -> tuple[str, int | None]:
    """Run a single salnet_symm.py job.

    Returns:
        A (label, returncode) tuple. returncode is None if the run was skipped.
    """
    label = f"lr={lr}/seed_{seed}"
    if (run_dir / "metrics.json").exists():
        return label, None  # already done
    result = subprocess.run(
        [
            sys.executable,
            "salnet_symm.py",
            "--seed",
            str(seed),
            "--run-dir",
            str(run_dir),
            "--lr",
            str(lr),
            "--n_epochs",
            str(n_epochs),
            "--len_epoch",
            str(len_epoch),
            "--batchsize",
            str(batchsize),
        ],
        check=False,
    )
    return label, result.returncode


def main() -> None:
    args = parse_args()
    sweep_dir = Path(args.base_dir) / args.sweep_name
    total = len(args.lrs) * args.n_seeds

    print(f"Sweep: {total} runs → {sweep_dir}")
    print(f"  lrs: {args.lrs}")
    print(f"  seeds: 0 … {args.n_seeds - 1}")
    print(f"  workers: {args.n_workers}")
    print()

    runs = [
        (sweep_dir / f"lr_{lr}" / f"seed_{seed}", lr, seed)
        for lr in args.lrs
        for seed in range(args.n_seeds)
    ]

    done = 0
    with ThreadPoolExecutor(max_workers=args.n_workers) as pool:
        futures = {
            pool.submit(
                _run_one,
                run_dir,
                lr,
                seed,
                args.n_epochs,
                args.len_epoch,
                args.batchsize,
            ): i
            for i, (run_dir, lr, seed) in enumerate(runs)
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
