#!/usr/bin/env python3
"""Generate a jobs.sh file for SLURM array job submission.

This is step 1 of the SLURM workflow for reproducing the SymmNet paper figure.
Each line of the output file contains one `python main_salnet.py ...` command,
one per dataset × algo × seed combination. Already-completed runs (those with a
metrics.json in the expected output directory) are skipped.

Full workflow
-------------
1. Generate the job list::

       python sweep_creator.py [--datasets ...] [--algos ...] [--n-seeds N]

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

ALL_DATASETS = ["cifar10", "fmnist", "svhn"]
ALL_ALGOS = ["bp", "fa", "bp_w_fa", "akrout", "scfa", "sal", "rdd"]
PARAM_FILE = "exp_settings.yaml"
PY_FILE = "main_salnet.py"
JOBS_FILE = "jobs.sh"


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


def main() -> None:
    args = parse_args()
    sweep_dir = Path(args.base_dir) / args.sweep_name
    total = len(args.datasets) * len(args.algos) * args.n_seeds
    done = 0

    print(f"Sweep: {total} runs → {sweep_dir}")
    print(f"  datasets: {args.datasets}")
    print(f"  algos: {args.algos}")
    print(f"  seeds: 0 … {args.n_seeds - 1}")
    print()

    if Path(JOBS_FILE).exists():
        print(f"{JOBS_FILE} already exists. Stop here!")
        return

    proc_calls = []
    for dataset in args.datasets:
        for algo in args.algos:
            for seed in range(args.n_seeds):
                run_dir = sweep_dir / dataset / algo / f"seed_{seed}"
                done += 1

                if (run_dir / "metrics.json").exists():
                    print(f"[{done}/{total}] Skip  {dataset}/{algo}/seed_{seed}")
                    continue

                print(f"[{done}/{total}] Run   {dataset}/{algo}/seed_{seed}")
                proc_calls.append(
                    (
                        f"python {PY_FILE} -f {PARAM_FILE} -s {algo}"
                        f" --dataset {dataset} --seed {seed} --run-dir {run_dir}\n"
                    )
                )

    with open(JOBS_FILE, "w") as f:
        f.writelines(proc_calls)


if __name__ == "__main__":
    main()
