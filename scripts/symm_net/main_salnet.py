#!/usr/bin/env python3

import json
import os
import shutil
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np  # noqa
import torch
from tqdm import tqdm

from load_utils import settings_loader
from symmnet import ConvNet
from symmnet.datasets import cifar10, fmnist, mnist, svhn  # noqa
from symmnet.sal_net import SALNet
from symmnet.utils import asym_angle, asym_var, corrcoef

# ---------------------------
# Intervals
# ---------------------------

CHECKPOINT_INTERVAL = 10
SCATTER_INTERVAL = 10

# ---------------------------
# All relevant settings (standard)
# ---------------------------

params = {
    "n_epochs": 25,
    "batch_size": 64,
    "lr": 0.01,
    "momentum": 0.9,
    "weight_decay": 0.0,
    "use_backprop": False,
    "use_kp": False,
    "use_fa_conv_layers": False,
    "use_scfa": False,
}

sal_params = {
    "n_iterations": 5,
    "use_sal": False,
    "t_ref": 10,
    "len_epoch": 200,
    "sal_lr": 0.05,
    "batch_size": 16,
}

rdd_params = {
    "rdd_time": 90,
    "use_rdd": True,
    "every_epoch": True,
}

# choose the dataset:
dataset = cifar10

# ---------------------------
# Argument parsing and param file loading
# ---------------------------

(
    params,
    sal_params,
    rdd_params,
    dataset,
    tags,
    group_tags,
    param_file,
    output_dir,
) = settings_loader()

# some general checks:
assert not (params["use_backprop"] and sal_params["use_sal"])
assert not (params["use_backprop"] and rdd_params["use_rdd"])
assert not (sal_params["use_sal"] and rdd_params["use_rdd"])

# ---------------------------
# Data tracking directory utilities
# ---------------------------


def create_run_dirs(base_dir="runs", tags=None):
    """Create a timestamped run directory with subfolders for figs and checkpoints."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag_str = "_".join(tags) if tags else "run"
    run_dir = os.path.join(base_dir, f"{timestamp}_{tag_str}")

    os.makedirs(os.path.join(run_dir, "figs", "weights"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)

    return run_dir


def save_json(obj, path):
    """Serialize obj to a JSON file at path."""
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def append_metric(metrics_dict, key, value):
    """Append a scalar value to a list under key in metrics_dict."""
    if key not in metrics_dict:
        metrics_dict[key] = []
    metrics_dict[key].append(float(value))


# ---------------------------
# Data tracking setup
# ---------------------------

run_dir = create_run_dirs(base_dir=output_dir, tags=tags)

# copy param file to run root
if param_file is not None:
    shutil.copy(param_file, os.path.join(run_dir, os.path.basename(param_file)))

# in-memory metrics store
metrics = {
    "params": params,
    "sal_params": sal_params,
    "rdd_params": rdd_params,
    "dataset": dataset.__name__,
    "scalars": {},
}

# ---------------------------
# Data Loading and Transforms
# ---------------------------

train_loader, test_loader, n_channels, shape = dataset(params["batch_size"])

# ---------------------------
# Model, Loss, and Optimizer
# ---------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
n_layers = 4

net = ConvNet(
    input_channels=n_channels,
    input_shape=shape,
    use_backprop=params["use_backprop"],
    use_kp=params["use_kp"],
    use_fa_conv_layers=params["use_fa_conv_layers"],
    use_scfa=params["use_scfa"],
).to(device)

loss_fn = torch.nn.CrossEntropyLoss()

if params["use_kp"]:
    optimizer = torch.optim.SGD(
        [
            {"params": net.parameters_weight(), "weight_decay": params["weight_decay"]},
            {
                "params": net.parameters_fb_weight(),
                "weight_decay": params["weight_decay"],
            },
            {"params": net.parameters_other()},
        ],
        lr=params["lr"],
        momentum=params["momentum"],
    )
else:
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=params["lr"],
        momentum=params["momentum"],
    )

if sal_params["use_sal"]:
    sal_net = SALNet(
        net.feature_layers_sizes,
        sal_params["t_ref"],
        sal_params["sal_lr"],
        batch_size=sal_params["batch_size"],
        buffer_length=2,
    )
    sal_net.to(device)
    sal_net.load_common_state_dict(net.state_dict())

if rdd_params["use_rdd"]:
    from symmnet import RDDNet, dt, input_rate, mem

    rdd_net = RDDNet(net.feature_layers_sizes)
    symm_losses = [[] for _ in range(3)]
    decay_losses = [[] for _ in range(3)]
    sparse_losses = [[] for _ in range(3)]
    self_losses = [[] for _ in range(3)]
    amp_losses = [[] for _ in range(3)]
    info_losses = [[] for _ in range(3)]
    corr_percents = [[] for _ in range(3)]

# ---------------------------
# Training and Evaluation Functions
# ---------------------------


def train(model, optimizer, criterion, train_loader, device, metrics):
    """Standard training loop for one epoch."""
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader)
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss_val = criterion(outputs, targets)
        loss_val.backward()
        optimizer.step()

        train_loss += loss_val.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar.set_description(
            "Train Loss: {:.3f} | Acc: {:.3f}% ({:d}/{:d})".format(
                train_loss / (batch_idx + 1), 100 * correct / total, correct, total
            )
        )

    append_metric(metrics["scalars"], "loss/train", train_loss)
    append_metric(metrics["scalars"], "accuracy/train", 100 * correct / total)


def test(model, criterion, test_loader, device, metrics):
    """Evaluation loop on the test set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(test_loader)
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss_val = criterion(outputs, targets)

            total_loss += loss_val.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar.set_description(
                "Test Loss: {:.3f} | Acc: {:.3f}% ({:d}/{:d})".format(
                    total_loss / (batch_idx + 1), 100 * correct / total, correct, total
                )
            )

    append_metric(metrics["scalars"], "loss/test", total_loss)
    append_metric(metrics["scalars"], "accuracy/test", 100 * correct / total)


def train_sal(sal_net, sal_params):
    """Run one SAL phase (spike-based feedback weight update)."""
    progress_bar = tqdm(range(sal_params["len_epoch"] * sal_params["t_ref"]))
    with torch.no_grad():
        for _ in progress_bar:
            sal_net.update_mempot()
            sal_net.update_spikes()
            sal_net.fb_stdp_online()
        dw = sal_net.apply_fb_weight_update()
    return dw


def eval_symmetry(net, metrics, run_dir, epoch):
    """Evaluate forward/feedback weight symmetry; save scatter plots every SCATTER_INTERVAL epochs."""
    weights = list(net.parameters_weight())
    fb_weights = list(net.parameters_fb_weight(ignore_require_grad=True))

    for i in range(n_layers - 1):
        w = weights[i].detach()
        b = weights[i].detach() if len(fb_weights) == 0 else fb_weights[i].detach()

        angle = asym_angle(w, b)
        var = asym_var(w, b)
        corr = corrcoef(w, b)
        sign = (torch.sign(w) * torch.sign(b)).flatten().mean()

        append_metric(metrics["scalars"], f"symm/angle/{i}", angle)
        append_metric(metrics["scalars"], f"symm/var/{i}", var)
        append_metric(metrics["scalars"], f"symm/corrcoef/{i}", corr)
        append_metric(metrics["scalars"], f"symm/sign/{i}", sign)

        if epoch % SCATTER_INTERVAL == 0:
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

            fig_path = os.path.join(layer_dir, f"scatter_{epoch:03d}.png")  # noqa
            fig.savefig(fig_path, bbox_inches="tight")
            plt.close(fig)


def save_final_plots(metrics, run_dir):
    """Generate and save final summary plots for loss, accuracy, and symmetry angle."""
    scalars = metrics["scalars"]
    figs_dir = os.path.join(run_dir, "figs")

    # loss
    fig, ax = plt.subplots()
    ax.plot(scalars.get("loss/train", []), label="train")
    ax.plot(scalars.get("loss/test", []), label="test")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid()
    fig.savefig(os.path.join(figs_dir, "loss.png"), bbox_inches="tight")
    plt.close(fig)

    # accuracy
    fig, ax = plt.subplots()
    ax.plot(scalars.get("accuracy/train", []), label="train")
    ax.plot(scalars.get("accuracy/test", []), label="test")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()
    ax.grid()
    fig.savefig(os.path.join(figs_dir, "accuracy.png"), bbox_inches="tight")
    plt.close(fig)

    # symmetry angle per layer
    fig, ax = plt.subplots()
    for i in range(n_layers - 1):
        key = f"symm/angle/{i}"
        if key in scalars:
            ax.plot(scalars[key], label=f"layer_{i}")
    ax.set_xlabel("Eval step")
    ax.set_ylabel("Asymmetry angle")
    ax.legend()
    ax.grid()
    fig.savefig(os.path.join(figs_dir, "angle.png"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------
# RDD Training
# ---------------------------


def train_rdd():
    """Implement the spike-based RDD feed-back learning."""
    rdd_net.reset()

    rdd_net.copy_weights_from([net.conv1, net.conv2, net.fc1, net.fc2, net.fc3])

    RDD_time = int(rdd_params["rdd_time"] / dt)

    print("Performing RDD pre-training for {} s...".format(rdd_params["rdd_time"]))

    for i in range(3):
        weight = rdd_net.classification_layers[-3 + i].weight_orig
        fb_weight = rdd_net.classification_layers[-4 + i].fb_weight
        x = np.random.uniform(0, 1, size=(weight.shape[1], 1))

        if np.sum(fb_weight != 0) > 0:
            corr_percent = (
                100
                * np.sum((np.sign(fb_weight.T) == np.sign(weight)) & (fb_weight.T != 0))
                / np.sum(fb_weight.T != 0)
            )
        else:
            corr_percent = 0

        corr_percents[i].append(corr_percent)

        decay_loss = (
            np.linalg.norm(np.dot(weight, x)) ** 2
            + np.linalg.norm(np.dot(x.T, fb_weight)) ** 2
        )
        sparse_loss = np.linalg.norm(weight) ** 2 + np.linalg.norm(fb_weight) ** 2
        self_loss = -np.trace(np.dot(fb_weight, weight))
        amp_loss = -np.trace(np.dot(x.T, np.dot(fb_weight, np.dot(weight, x))))
        info_loss = np.linalg.norm(x - np.dot(fb_weight, np.dot(weight, x))) ** 2
        symm_loss = np.linalg.norm(weight - fb_weight.T) ** 2

        decay_losses[i].append(decay_loss)
        sparse_losses[i].append(sparse_loss)
        self_losses[i].append(self_loss)
        amp_losses[i].append(amp_loss)
        info_losses[i].append(info_loss)
        symm_losses[i].append(symm_loss)

    text = (
        f"Time: {0 * dt}/{RDD_time * dt} s. "
        f"Correct: {corr_percents[0][-1]:.2f}% / {corr_percents[1][-1]:.2f}% / {corr_percents[2][-1]:.2f}%. "  # noqa
        f"Trace:  {self_losses[0][-1]:.2f} / {self_losses[1][-1]:.2f} / {self_losses[2][-1]:.2f}."  # noqa
    )

    print(text)

    indices_1 = np.random.choice(
        net.feature_layers_sizes[0],
        int(0.2 * net.feature_layers_sizes[0]),
        replace=False,
    )
    indices_2 = np.random.choice(384, int(0.2 * 384), replace=False)
    indices_3 = np.random.choice(192, int(0.2 * 192), replace=False)

    driving_rates_1 = np.zeros((net.feature_layers_sizes[0], 1))
    driving_rates_2 = np.zeros((384, 1))
    driving_rates_3 = np.zeros((192, 1))

    driving_rates_1[indices_1] = input_rate
    driving_rates_2[indices_2] = input_rate
    driving_rates_3[indices_3] = input_rate

    driving_spike_hist_1 = np.zeros((net.feature_layers_sizes[0], mem), dtype=int)
    driving_spike_hist_2 = np.zeros((384, mem), dtype=int)
    driving_spike_hist_3 = np.zeros((192, mem), dtype=int)

    spike_rates_1 = np.zeros(net.feature_layers_sizes[0])
    spike_rates_2 = np.zeros(384)
    spike_rates_3 = np.zeros(192)
    spike_rates_4 = np.zeros(10)

    for i in range(RDD_time):
        if (i + 1) % (0.1 / dt) == 0:
            indices_1 = np.random.choice(
                net.feature_layers_sizes[0],
                int(0.2 * net.feature_layers_sizes[0]),
                replace=False,
            )
            indices_2 = np.random.choice(384, int(0.2 * 384), replace=False)
            indices_3 = np.random.choice(192, int(0.2 * 192), replace=False)

            driving_rates_1 = np.zeros((net.feature_layers_sizes[0], 1))
            driving_rates_2 = np.zeros((384, 1))
            driving_rates_3 = np.zeros((192, 1))

            driving_rates_1[indices_1] = input_rate
            driving_rates_2[indices_2] = input_rate
            driving_rates_3[indices_3] = input_rate

        if i < RDD_time / 3:
            driving_spike_hist_1 = np.concatenate(
                [driving_spike_hist_1[:, 1:], np.random.poisson(driving_rates_1)],
                axis=-1,
            )
        elif i < 2 * RDD_time / 3:
            driving_spike_hist_2 = np.concatenate(
                [driving_spike_hist_2[:, 1:], np.random.poisson(driving_rates_2)],
                axis=-1,
            )
        else:
            driving_spike_hist_3 = np.concatenate(
                [driving_spike_hist_3[:, 1:], np.random.poisson(driving_rates_3)],
                axis=-1,
            )

        if i < RDD_time / 3:
            rdd_net.out(driving_spike_hist_1, None, None)
        elif i < 2 * RDD_time / 3:
            rdd_net.out(None, driving_spike_hist_2, None)
        else:
            rdd_net.out(None, None, driving_spike_hist_3)

        spike_rates_1 += rdd_net.classification_layers[0].spike_hist[:, -1]
        spike_rates_2 += rdd_net.classification_layers[1].spike_hist[:, -1]
        spike_rates_3 += rdd_net.classification_layers[2].spike_hist[:, -1]
        spike_rates_4 += rdd_net.classification_layers[3].spike_hist[:, -1]

        if (i + 1) % (10.0 / dt) == 0:
            rdd_net.update_fb_weights()

            spike_rates_1 /= 10
            spike_rates_2 /= 10
            spike_rates_3 /= 10
            spike_rates_4 /= 10

            for j in range(3):
                weight = rdd_net.classification_layers[-3 + j].weight_orig
                fb_weight = rdd_net.classification_layers[-4 + j].fb_weight
                beta = rdd_net.classification_layers[-4 + j].beta  # noqa
                x = np.random.uniform(0, 1, size=(weight.shape[1], 1))

                if np.sum(fb_weight != 0) > 0:
                    corr_percent = (
                        100
                        * np.sum(
                            (np.sign(fb_weight.T) == np.sign(weight))
                            & (fb_weight.T != 0)
                        )
                        / np.sum(fb_weight.T != 0)
                    )
                else:
                    corr_percent = 0

                corr_percents[j].append(corr_percent)

                decay_loss = (
                    np.linalg.norm(np.dot(weight, x)) ** 2
                    + np.linalg.norm(np.dot(x.T, fb_weight)) ** 2
                )
                sparse_loss = (
                    np.linalg.norm(weight) ** 2 + np.linalg.norm(fb_weight) ** 2
                )
                self_loss = -np.trace(np.dot(fb_weight, weight))
                amp_loss = -np.trace(np.dot(x.T, np.dot(fb_weight, np.dot(weight, x))))
                info_loss = (
                    np.linalg.norm(x - np.dot(fb_weight, np.dot(weight, x))) ** 2
                )
                symm_loss = np.linalg.norm(weight - fb_weight.T) ** 2

                decay_losses[j].append(decay_loss)
                sparse_losses[j].append(sparse_loss)
                self_losses[j].append(self_loss)
                amp_losses[j].append(amp_loss)
                info_losses[j].append(info_loss)
                symm_losses[j].append(symm_loss)

            text = "Time: {}/{} s. Correct: {:.2f}% / {:.2f}% / {:.2f}%. Trace: {:.2f} / {:.2f} / {:.2f}. Rates: {:.2f}Hz / {:.2f}Hz / {:.2f}Hz / {:.2f}Hz. ".format(
                (i + 1) * dt,
                RDD_time * dt,
                corr_percents[0][-1],
                corr_percents[1][-1],
                corr_percents[2][-1],
                self_losses[0][-1],
                self_losses[1][-1],
                self_losses[2][-1],
                np.mean(spike_rates_1),
                np.mean(spike_rates_2),
                np.mean(spike_rates_3),
                np.mean(spike_rates_4),
            )

            print(text)

            spike_rates_1 *= 0
            spike_rates_2 *= 0
            spike_rates_3 *= 0
            spike_rates_4 *= 0

    rdd_net.copy_weights_to([net.conv1, net.conv2, net.fc1, net.fc2, net.fc3], device)


# ---------------------------
# Main Training Loop
# ---------------------------


def main():
    eval_symmetry(net, metrics, run_dir, epoch=0)

    for epoch in range(params["n_epochs"]):
        print(f"Epoch {epoch}/{params['n_epochs']}.")

        if sal_params["use_sal"]:
            print("SAL phase.")
            sal_net.load_common_state_dict(net.state_dict())
            for _ in range(sal_params["n_iterations"]):
                train_sal(sal_net, sal_params)
            net.load_state_dict(sal_net.get_common_state_dict(), strict=False)
            eval_symmetry(net, metrics, run_dir, epoch)

        if rdd_params["use_rdd"]:
            print("RDD phase.")
            train_rdd()
            eval_symmetry(net, metrics, run_dir, epoch)

        train(net, optimizer, loss_fn, train_loader, device, metrics)
        test(net, loss_fn, test_loader, device, metrics)

        if not (sal_params["use_sal"] or rdd_params["use_rdd"]):
            eval_symmetry(net, metrics, run_dir, epoch)

        if epoch % CHECKPOINT_INTERVAL == 0:
            ckpt_path = os.path.join(
                run_dir, "checkpoints", f"epoch_{epoch:03d}.pt"  # noqa
            )
            torch.save(net.state_dict(), ckpt_path)
            print("Saved checkpoint:", ckpt_path)

        save_json(metrics, os.path.join(run_dir, "metrics.json"))
        plt.close("all")

    torch.save(net.state_dict(), os.path.join(run_dir, "checkpoints", "final_model.pt"))
    save_json(metrics, os.path.join(run_dir, "metrics.json"))
    save_final_plots(metrics, run_dir)
    print("Run complete. Results saved to:", run_dir)


if __name__ == "__main__":
    main()
