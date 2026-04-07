#!/usr/bin/env python3

import argparse

import yaml

from symmnet.datasets import cifar10, fmnist, mnist, svhn  # noqa

# Hardcoded defaults
DEFAULT_PARAMS = {
    "params": {
        "n_epochs": 100,
        "batch_size": 64,
        "lr": 0.01,
        "momentum": 0.9,
        "weight_decay": 0.0,
        "use_backprop": True,
        "use_kp": False,
        "use_fa_conv_layers": False,
        "use_scfa": False,
    },
    "sal_params": {
        "n_iterations": 5,
        "use_sal": False,
        "t_ref": 10,
        "len_epoch": 200,
        "sal_lr": 0.05,
        "batch_size": 16,
    },
    "rdd_params": {
        "rdd_time": 90,
        "use_rdd": False,
        "every_epoch": True,
    },
    "dataset": "cifar10",
}
ALLOWED_DATASETS = {"cifar10": cifar10, "mnist": mnist, "fmnist": fmnist, "svhn": svhn}


def parse_tags(s):
    return [tag.strip() for tag in s.split(",")] if s else []


def load_params(param_file, section=None):
    with open(param_file, "r") as f:
        content = yaml.safe_load(f)
    if section:
        if section not in content:
            raise ValueError(
                f"Section '{section}' not found in parameter file: {param_file}"  # noqa E713
            )
        block = content.get(section, {})
    else:
        # fallback to the first section
        block = next(val for val in content.values() if isinstance(val, dict))
    return (
        block.get("params", {}),
        block.get("sal_params", {}),
        block.get("rdd_params", {}),
        block.get("dataset", "mnist"),
    )


def merge(base, override):
    result = base.copy()
    result.update(override)
    return result


def settings_loader():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", type=str, help="Path to parameter file.")
    parser.add_argument("-s", type=str, help="Section name in YAML file.")
    parser.add_argument("-i", type=int, default=0, help="Slurm run id.")
    parser.add_argument("--dataset", type=str, choices=ALLOWED_DATASETS)
    parser.add_argument("--n_epochs", type=int)
    parser.add_argument("--tags", type=str)
    parser.add_argument("--group_tags", type=str)
    parser.add_argument(
        "--output-dir", type=str, default="../../results/symm_net", dest="output_dir"
    )
    args = parser.parse_args()

    if args.f:
        (
            params_from_file,
            sal_params_from_file,
            rdd_params_from_file,
            dataset_from_file,
        ) = load_params(args.f, args.s)
    else:
        (
            params_from_file,
            sal_params_from_file,
            rdd_params_from_file,
            dataset_from_file,
        ) = ({}, {}, {}, "mnist")

    # Precedence: default < YAML < command line
    params = merge(DEFAULT_PARAMS["params"], params_from_file)
    sal_params = merge(DEFAULT_PARAMS["sal_params"], sal_params_from_file)
    rdd_params = merge(DEFAULT_PARAMS["rdd_params"], rdd_params_from_file)
    dataset_name = args.dataset or dataset_from_file or DEFAULT_PARAMS["dataset"]
    params["n_epochs"] = args.n_epochs or params.get("n_epochs", 25)
    tags_list = parse_tags(args.tags) if args.tags else []
    group_tags_list = parse_tags(args.group_tags) if args.group_tags else []

    if dataset_name in ALLOWED_DATASETS:
        dataset = ALLOWED_DATASETS[dataset_name]
    else:
        raise ValueError(f"Dataset '{dataset_name}' not allowed")

    return (
        params,
        sal_params,
        rdd_params,
        dataset,
        tags_list,
        group_tags_list,
        args.f,
        args.output_dir,
    )
