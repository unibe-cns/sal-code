#!/usr/bin/env python3

"""Train a fully connected BM"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from neuralsampling import utils
from neuralsampling.network import GradDescent, NeuralSamplerFullyConnected, rect_kernel
from neuralsampling.stdp_functions import STDPRuler

paramfile_name = sys.argv[1]
paramfile_stem = Path(paramfile_name).with_suffix("")
params = utils.load_paramfile(paramfile_name)

t_rng = np.random.default_rng(params["seed"]["distr"])
i_rng = np.random.default_rng(params["seed"]["init"])

# target and initial parameters:
a = 0.5
b = 0.5
w_target = (
    2
    * t_rng.beta(
        params["distr"]["alpha"],
        params["distr"]["beta"],
        size=(params["dim"], params["dim"]),
    )
    - 1
)
w_target = utils.copy_triu(w_target)
b_target = (
    2
    * t_rng.beta(params["distr"]["alpha"], params["distr"]["beta"], size=params["dim"])
    - 1
)
print("B TARGET", b_target)

w_init = i_rng.normal(0.0, params["weight_init"], size=(params["dim"], params["dim"]))
w_init = utils.copy_triu(w_init)
w_init = w_init + i_rng.normal(
    0.0, params["init_noise"], size=(params["dim"], params["dim"])
)
b_init = i_rng.normal(0.0, params["bias_init"], params["dim"])

params["sim_params"]["psp_kernel"] = rect_kernel

stdp_rule = STDPRuler.tri_kernel(
    params["dim"],
    params["stdp"]["num_last_spikes"],
    params["stdp"]["ws"]["a_plus"],
    params["stdp"]["ws"]["a_minus"],
    params["stdp"]["ws"]["tau_plus"],
    params["stdp"]["ws"]["tau_minus"],
)
# atm use only left-right / wake-sleep symmetric kernels
a_forward = utils.draw_trunc_distr(
    i_rng.normal,
    high=params["stdp"]["ws"]["noise"] * 1.5,
    low=-params["stdp"]["ws"]["noise"] * 1.5,
    size=(params["dim"], params["dim"]),
    kwargs={"loc": 0.0, "scale": params["stdp"]["ws"]["noise"]},
)
a_forward = a_forward + params["stdp"]["ws"]["a_plus"]

a_backward = utils.draw_trunc_distr(
    i_rng.normal,
    high=params["stdp"]["ws"]["noise"] * 1.5,
    low=-params["stdp"]["ws"]["noise"] * 1.5,
    size=(params["dim"], params["dim"]),
    kwargs={"loc": 0.0, "scale": params["stdp"]["ws"]["noise"]},
)
a_backward = a_backward + params["stdp"]["ws"]["a_plus"]

stdp_rule.set_forward(a_plus=a_forward, a_minus=a_forward)
stdp_rule.set_backward(a_plus=a_backward, a_minus=a_backward)

stdp_rule_sal = STDPRuler.tri_kernel(
    params["dim"],
    params["stdp"]["num_last_spikes"],
    params["stdp"]["sal"]["a_plus"],
    params["stdp"]["sal"]["a_minus"],
    params["stdp"]["sal"]["tau_plus"],
    params["stdp"]["sal"]["tau_minus"],
)
a_forward = utils.draw_trunc_distr(
    i_rng.normal,
    high=params["stdp"]["ws"]["noise"] * 1.5,
    low=-params["stdp"]["ws"]["noise"] * 1.5,
    size=(params["dim"], params["dim"]),
    kwargs={"loc": 0.0, "scale": params["stdp"]["ws"]["noise"]},
)
a_forward = a_forward + params["stdp"]["ws"]["a_plus"]

a_backward = utils.draw_trunc_distr(
    i_rng.normal,
    high=params["stdp"]["ws"]["noise"] * 1.5,
    low=-params["stdp"]["ws"]["noise"] * 1.5,
    size=(params["dim"], params["dim"]),
    kwargs={"loc": 0.0, "scale": params["stdp"]["ws"]["noise"]},
)
a_backward = a_backward + params["stdp"]["ws"]["a_plus"]
stdp_rule_sal.set_forward(a_plus=-a_forward, a_minus=a_forward)
stdp_rule_sal.set_backward(a_plus=-a_backward, a_minus=a_backward)


opt_bias = GradDescent(params["lr"]["bias"])
opt_weight = GradDescent(params["lr"]["weight"])
opt_symm = GradDescent(params["lr"]["symm"])


sampler = NeuralSamplerFullyConnected(
    w_init,
    b_init,
    w_target,
    b_target,
    params["sim_params"],
    params["dur"] * params["sim_params"]["t_ref"],
    opt_bias,
    opt_weight,
    opt_symm,
    max_w=params["max_w"],
    max_b=params["max_b"],
    rng_seed=params["seed"]["sim"],
)

all_dkl = []
all_weights = []
all_biases = []
all_asym = []
all_distr = []


def callback(res, step):
    if sampler.validation:
        distr = res["sampled_distr"]
        current_dkl = utils.calc_dkl(res["target_distr"], distr)
        weights = res["weights"] - res["weights"].T
        upper_half = weights[np.triu_indices(weights.shape[0], k=1)]
        asym = np.var(upper_half)
        print(f"ASYM {asym:.5f}")
        print(f"DKL {current_dkl:.5f}")
        all_distr.append(res["sampled_distr"])
        all_dkl.append(current_dkl)
        all_biases.append(res["biases"])
        all_asym.append(asym)
        all_weights.append(res["weights"])


sampler.train(
    params["num_epochs"],
    stdp_rule,
    stdp_rule_sal,
    callback,
    validation_step=params["val_step"],
    validation_factor=params["val_factor"],
)


fig, ax = plt.subplots(6, 1, figsize=(6, 10))
ax[0].plot(all_dkl)
ax[0].set_ylabel("DKL")
ax[0].set_yscale("log")
ax[0].set_title(f'init_noise {params["init_noise"]}')
ax[1].plot(all_asym)
ax[1].set_ylabel("weight asymmetry")
all_weights_trimmed = np.array([w[w != 0.0] for w in all_weights])
ax[2].plot(all_weights_trimmed, color="C0", alpha=0.1)
[ax[2].axhline(y=w_t, color="k", alpha=0.2, ls="--") for w_t in np.nditer(w_target)]
ax[2].set_ylabel("weights")
ax[3].plot(np.array(all_biases), color="C3", alpha=0.1)
[ax[3].axhline(y=b_t, color="k", alpha=0.2, ls="--") for b_t in np.nditer(b_target)]
ax[3].set_ylabel("biases")
weight_diffs = [w - w.T for w in all_weights]
weight_diffs = np.array([w[np.triu_indices_from(w)] for w in weight_diffs])
ax[4].plot(weight_diffs, color="C2", alpha=0.1)
ax[4].set_ylabel("weight diffs")
ax[4].set_xlabel("epochs")
utils.plot_distr(
    ax[5], [sampler.target_distr, all_distr[-1]], sampler.los, ["target", "sampled"]
)

plt.tight_layout()

fig.savefig(f"{paramfile_stem}.png", dpi=300)


np.savez_compressed(
    f"{paramfile_stem}",
    dkls=np.array(all_dkl, dtype=np.float32),
    target_distr=np.array(sampler.target_distr, dtype=np.float32),
    all_distr=np.array(all_distr, dtype=np.float32),
    all_weights=np.array(all_weights, dtype=np.float32),
    all_biases=np.array(all_biases, dtype=np.float32),
    all_asym=np.array(all_asym, dtype=np.float32),
)
