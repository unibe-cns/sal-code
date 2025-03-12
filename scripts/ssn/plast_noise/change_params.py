"""Cerate individual experiment files for a parameter sweep."""

from itertools import product
from pathlib import Path

import yaml

FNAME = Path("exp.yaml")

# path  to the result files:
RES_PATH = Path("../../../results/ssn/plast_noise")
RES_PATH.mkdir(parents=True, exist_ok=True)

with open(FNAME, "r") as f:
    data = yaml.safe_load(f)

STDP_NOISE = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
DISTR_SEEDS = [4321, 4322, 4323, 4324, 4235]
INIT_SEEDS = [9002, 9003, 9004, 9005]
SIM_SEED = 7654
for i, (noise, d_seed, i_seed) in enumerate(
    product(STDP_NOISE, DISTR_SEEDS, INIT_SEEDS)
):
    data["stdp"]["ws"]["noise"] = noise
    data["stdp"]["sal"]["noise"] = noise
    data["seed"]["distr"] = d_seed
    data["seed"]["init"] = i_seed
    data["seed"]["sim"] = SIM_SEED + i
    filename = f"{FNAME.stem}.{i:04d}.yaml"
    with open(RES_PATH / filename, "w") as f:
        yaml.dump(data, f)

with open("num_sims.txt", "w") as f:
    f.write(str(i) + "\n")

with open("res_path.txt", "w") as f:
    f.write(str(RES_PATH) + "\n")

print(f"change_params.py created {i+1} files at {RES_PATH}.")
