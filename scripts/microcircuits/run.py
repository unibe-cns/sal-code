#!/usr/bin/env python3

import argparse
import pickle
from pathlib import Path

from microcircuits.experiment import ExperimentDescriptor, run_student, run_teacher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file", help="Path to the experiment configuration file."
    )
    parser.add_argument(
        "proc_id",
        help="Process id for parallel execution",
        type=int,
        nargs="?",
        default=None,
    )
    args = parser.parse_args()
    config_filename = Path(args.config_file)
    fname_stem = Path(config_filename).parent

    teacher_plot_name = fname_stem / f"teacher.{args.proc_id:04d}.png"
    student_pickle_name = fname_stem / f"student.{args.proc_id:04d}.pickle"
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
