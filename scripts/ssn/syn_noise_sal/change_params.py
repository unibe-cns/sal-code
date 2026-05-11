"""Create individual experiment files for a parameter sweep."""

from itertools import product
from pathlib import Path

import yaml

FNAME = Path("exp.yaml")

with open(FNAME, "r") as f:
    data = yaml.safe_load(f)

INIT_NOISE = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
for i, (noise,) in enumerate(product(INIT_NOISE)):
    data["init_noise"] = noise
    filename = f"{FNAME.stem}.{i:04d}.yaml"
    with open(filename, "w") as f:
        yaml.dump(data, f)
