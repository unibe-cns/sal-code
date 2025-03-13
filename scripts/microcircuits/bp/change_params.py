"""Create individual experiment files for a parameter sweep."""

from pathlib import Path

import yaml

FNAME = Path("backprop.yaml")

# path  to the result files:
RES_PATH = Path("../../../results/microcircuits/bp")
RES_PATH.mkdir(parents=True, exist_ok=True)

with open(FNAME, "r") as f:
    data = yaml.safe_load(f)

NUM_SIMS = 20
for i in range(NUM_SIMS):
    data["student_simulation_settings"]["weights_init_seed"] += 1
    data["student_simulation_settings"]["poisson_seed"] += 1
    data["student_simulation_settings"]["training_seed"] += 1
    data["teacher_simulation_settings"]["training_seed"] += 1
    filename = f"{FNAME.stem}.{i:04d}.yaml"
    with open(RES_PATH / filename, "w") as f:
        yaml.dump(data, f)

with open("num_sims.txt", "w") as f:
    f.write(str(i) + "\n")

with open("res_path.txt", "w") as f:
    f.write(str(RES_PATH) + "\n")

print(f"change_params.py created {i+1} files at {RES_PATH}.")
