#!/usr/bin/env python3

"""Generate a jobs.sh file for SLURM array job submission.

This is step 1 of the SLURM workflow for reproducing the SymmNet paper figure.
Each line of the output file contains one `python salnet_symm.py ...` command,
one per learning_rate × seed combination. Already-completed runs (those with a
metrics.json in the expected output directory) are skipped.

Full workflow
-------------
1. Generate the job list::

       python sweep_creator.py [--lrs ...] [--n-seeds N]

   This writes ``jobs.sh`` in the current directory.

2. Submit as a SLURM array job (snapshots jobs.sh first for reproducibility)::

       bash slurm_submit.sh

   Internally, slurm_submit.sh calls ``sbatch --array=1-N slurm.sh``.
   Each array task reads its own command from the snapshot and executes it.

3. After all jobs finish, run ``plots.ipynb`` to reproduce the figure.

Notes
-----
- ``jobs.sh`` must not exist before running this script (guard against
  accidental overwrites); delete it manually to regenerate.
- To run locally instead of on SLURM, use ``sweep.py`` directly.
"""

import argparse
from pathlib import Path

DEFAULT_LRS = [0.01, 0.02, 0.04, 0.08]
PY_FILE = "salnet_symm.py"
JOBS_FILE = "jobs.sh"


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


def main() -> None:
    args = parse_args()
    sweep_dir = Path(args.base_dir) / args.sweep_name
    total = len(args.lrs) * args.n_seeds
    done = 0

    print(f"Sweep: {total} runs → {sweep_dir}")
    print(f"  lrs: {args.lrs}")
    print(f"  seeds: 0 … {args.n_seeds - 1}")
    print(f"  workers: {args.n_workers}")
    print()

    if Path(JOBS_FILE).exists():
        print(f"{JOBS_FILE} already exists. Stop here!")
        return

    proc_calls = []
    for lr in args.lrs:
        for seed in range(args.n_seeds):
            run_dir = sweep_dir / f"lr_{lr}" / f"seed_{seed}"
            done += 1

            if (run_dir / "metrics.json").exists():
                print(f"[{done}/{total}] Skip  lr_{lr}/seed_{seed}")
                continue

            print(f"[{done}/{total}] Run   lr_{lr}/seed_{seed}")
            proc_calls.append(
                (
                    f"python {PY_FILE} --lr {lr} --seed {seed} --n_epochs {args.n_epochs} "
                    f"--len_epoch {args.len_epoch} --batchsize {args.batchsize} "
                    f"--run-dir {run_dir}\n"
                )
            )

    with open(JOBS_FILE, "w") as f:
        f.writelines(proc_calls)


if __name__ == "__main__":
    main()
