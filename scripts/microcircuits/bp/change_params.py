"""Create individual experiment files for a parameter sweep."""

from pathlib import Path

import yaml

FNAME = Path("exp.yaml")

with open(FNAME, "r") as f:
    data = yaml.safe_load(f)

NUM_SIMS = 20
for i in range(NUM_SIMS):
    data["student_simulation_settings"]["weights_init_seed"] += 1
    data["student_simulation_settings"]["poisson_seed"] += 1
    data["student_simulation_settings"]["training_seed"] += 1
    data["teacher_simulation_settings"]["training_seed"] += 1
    filename = f"{FNAME.stem}.{i:04d}.yaml"
    with open(filename, "w") as f:
        yaml.dump(data, f)

print(f"change_params.py created {i+1} files.")
