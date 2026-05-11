#!/usr/bin/env python3

"""Train a spiking microcircuit student network from a YAML experiment config.

This script implements the two-phase training procedure:

1. **Teacher phase** — runs a fixed teacher network (no weight updates) over the
   combined training and validation inputs to generate target membrane potentials.
   An input-vs-target scatter plot is saved alongside the results.

2. **Student phase** — trains a student network to reproduce the teacher's targets.
   Training is either done with "backprop", feedback alignment or SAL.
   The full simulation result dict is serialised to a pickle file for downstream
   analysis.

Usage::

    python train_mc.py <config_file> [proc_id] [-o <output_dir>]

The ``proc_id`` argument enables embarrassingly parallel execution: run the same
command with different ``proc_id`` values (e.g. for different random seeds encoded
in separate config files) and all outputs will land in the same directory without
collisions.
"""

import argparse
import pickle
import shutil
from pathlib import Path

from microcircuits.experiment import ExperimentDescriptor, run_student, run_teacher


def main():
    """Parse command-line arguments, run teacher and student, and save results.

    **CLI arguments**

    ``config_file``
        Path to the YAML experiment configuration file.  The file is copied into
        the output directory for reproducibility.

    ``proc_id`` (optional, default 0)
        Integer process ID used as a suffix in output file names, enabling
        multiple independent runs to write to the same directory without
        overwriting each other.

    ``-o`` (optional, default ``../../results/microcircuits``)
        Output directory.  Created automatically if it does not exist.

    **Output files**

    ``teacher.<proc_id:04d>.png``
        Scatter plot of teacher input vs. output membrane potentials for both
        training and validation sets.

    ``student.<proc_id:04d>.pickle``
        Pickled result dict from :func:`~microcircuits.experiment.run_student`,
        containing recorded quantities, spike trains, mean rates, and validation
        loss curves.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file", help="Path to the experiment configuration file."
    )
    parser.add_argument(
        "proc_id",
        help="Process id for parallel execution",
        type=int,
        nargs="?",
        default=0,
    )

    parser.add_argument(
        "-o",
        type=Path,
        default=Path("../../results/microcircuits"),
        help="Output directory (default: <repo_root>/results/ssn)",
    )

    args = parser.parse_args()
    config_filename = Path(args.config_file)
    outdir = args.o

    outdir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_filename, outdir)

    teacher_plot_name = outdir / f"teacher.{args.proc_id:04d}.png"
    student_pickle_name = outdir / f"student.{args.proc_id:04d}.pickle"
    exp = ExperimentDescriptor(config_filename)

    teacher_res, _ = run_teacher(
        exp.network_properties,
        exp.teacher_initial_parameters,
        exp.teacher_simulation_settings,
        exp.u_input,
        teacher_plot_name,
    )

    res = run_student(
        exp.network_properties,
        exp.student_initial_parameters,
        exp.student_simulation_settings,
        teacher_res,
    )

    with open(student_pickle_name, "wb") as f:
        pickle.dump(res, f)


if __name__ == "__main__":
    main()
