#!/usr/bin/env python3

import argparse
import json
import os
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from symmnet import CLASSIFICATION_LAYER_SIZES  # noqa
from symmnet.sal_net import SALNet
from symmnet.utils import asym_angle, asym_var, corrcoef

# ---------------------------
# Intervals
# ---------------------------

PLOT_INTERVAL = 50

# ---------------------------
# Default params
# ---------------------------

params = {
    "t_ref": 10,
    "n_epochs": 100,
    "len_epoch": 1_000,
    "sal_lr": 0.01,
    "batchsize": 1,
}
n_layers = 4

# ---------------------------
# Argument parsing
# ---------------------------


def parse_tags(s):
    return [tag.strip() for tag in s.split(",")] if s else []


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=params["n_epochs"])
parser.add_argument("--batchsize", type=int, default=params["batchsize"])
parser.add_argument("--len_epoch", type=int, default=params["len_epoch"])
parser.add_argument("--lr", type=float, default=params["sal_lr"])
parser.add_argument("--tags", type=str)
parser.add_argument("--group_tags", type=str)
args = parser.parse_args()

params["n_epochs"] = args.n_epochs
params["len_epoch"] = args.len_epoch
params["batchsize"] = args.batchsize
params["sal_lr"] = args.lr

tags = parse_tags(args.tags) if args.tags else []
group_tags = parse_tags(args.group_tags) if args.group_tags else []

# ---------------------------
# Run directory utilities
# ---------------------------


def create_run_dirs(base_dir="runs", tags=None):
    """Create a timestamped run directory with subfolders for figs."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag_str = "_".join(tags) if tags else "pure_sal"
    run_dir = os.path.join(base_dir, f"{timestamp}_{tag_str}")

    os.makedirs(os.path.join(run_dir, "figs", "weights"), exist_ok=True)

    return run_dir


def save_json(obj, path):
    """Serialize obj to a JSON file at path."""
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def append_metric(scalars, key, value):
    """Append a scalar value to a list under key in scalars."""
    if key not in scalars:
        scalars[key] = []
    scalars[key].append(float(value))


# ---------------------------
# Run setup
# ---------------------------

run_dir = create_run_dirs(base_dir="runs", tags=tags)

metrics = {
    "params": params,
    "scalars": {},
}

# ---------------------------
# Model
# ---------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

net = SALNet(
    CLASSIFICATION_LAYER_SIZES,
    params["t_ref"],
    params["sal_lr"],
    batch_size=params["batchsize"],
    buffer_length=2,
)
net.to(device)

# ---------------------------
# Helpers
# ---------------------------


def eval_and_log_symmetry(scalars, epoch):
    """Log symmetry metrics and save scatter plots every PLOT_INTERVAL epochs."""
    for i in range(n_layers - 1):
        w = net.layers[i + 1].weight.detach()
        b = net.layers[i].fb_weight.detach().t()

        append_metric(scalars, f"symm/angle/{i}", asym_angle(w, b))
        append_metric(scalars, f"symm/var/{i}", asym_var(w, b))
        append_metric(scalars, f"symm/corrcoef/{i}", corrcoef(w, b))

        if epoch % PLOT_INTERVAL == 0:
            layer_dir = os.path.join(run_dir, "figs", "weights", f"layer_{i}")
            os.makedirs(layer_dir, exist_ok=True)

            fig, ax = plt.subplots()
            w_np = w.cpu().numpy().flatten()
            b_np = b.cpu().numpy().flatten()
            boundary = max(np.max(np.abs(w_np)), np.max(np.abs(b_np))) * 1.1
            ax.scatter(w_np, b_np, alpha=0.1)
            ax.set_xlim(-boundary, boundary)
            ax.set_ylim(-boundary, boundary)
            ax.grid()
            ax.set_xlabel("W")
            ax.set_ylabel("B")
            ax.set_aspect("equal")
            fig.savefig(
                os.path.join(layer_dir, f"scatter_{epoch:03d}.png"),  # noqa
                bbox_inches="tight",
            )
            plt.close(fig)


def save_final_plots(metrics, run_dir):
    """Generate and save final summary plots for symmetry angle and corrcoef."""
    scalars = metrics["scalars"]
    figs_dir = os.path.join(run_dir, "figs")

    # angle per layer
    fig, ax = plt.subplots()
    for i in range(n_layers - 1):
        key = f"symm/angle/{i}"
        if key in scalars:
            ax.plot(scalars[key], label=f"layer_{i}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Asymmetry angle")
    ax.legend()
    ax.grid()
    fig.savefig(os.path.join(figs_dir, "angle.png"), bbox_inches="tight")
    plt.close(fig)

    # corrcoef per layer
    fig, ax = plt.subplots()
    for i in range(n_layers - 1):
        key = f"symm/corrcoef/{i}"
        if key in scalars:
            ax.plot(scalars[key], label=f"layer_{i}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Correlation coefficient (W, B)")
    ax.legend()
    ax.grid()
    fig.savefig(os.path.join(figs_dir, "corrcoef.png"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------
# Main
# ---------------------------


def main():
    # log symmetry before any training
    eval_and_log_symmetry(metrics["scalars"], epoch=0)
    save_json(metrics, os.path.join(run_dir, "metrics.json"))

    with torch.no_grad():
        for epoch in tqdm(range(params["n_epochs"])):
            for _ in range(params["len_epoch"] * params["t_ref"]):
                net.update_mempot()
                net.update_spikes()
                net.fb_stdp_online()

            net.apply_fb_weight_update()
            eval_and_log_symmetry(metrics["scalars"], epoch=epoch + 1)

            save_json(metrics, os.path.join(run_dir, "metrics.json"))
            plt.close("all")

    save_final_plots(metrics, run_dir)
    print("Run complete. Results saved to:", run_dir)


if __name__ == "__main__":
    main()
