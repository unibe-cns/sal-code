"""Create individual experiment files for a parameter sweep."""

from itertools import product
from pathlib import Path

import yaml


def set_value(dct, keys, val):
    for key in keys[:-1]:
        if isinstance(dct, dict) and key in dct:
            dct = dct[key]
        else:
            raise KeyError(f"{keys} doesn't exist!")
    if (
        isinstance(dct, dict)
        and keys[-1] in dct
        and not isinstance(dct[keys[-1]], dict)
    ):
        dct[keys[-1]] = val
    else:
        raise KeyError(f"{keys} doesn't exist!")


FNAME = Path("exp.yaml")

with open(FNAME, "r") as f:
    data = yaml.safe_load(f)

# LR_WS = [0.0005, 0.001, 0.002, 0.004, 0.008]
# LR_SAL = [0.0005, 0.001, 0.002, 0.004, 0.008]
STDP_NOISE = [0.0, 0.2, 0.4, 0.6, 0.8]
for i, (noise,) in enumerate(product(STDP_NOISE)):
    set_value(data, ("stdp", "ws", "noise"), noise)
    set_value(data, ("stdp", "sal", "noise"), noise)
    filename = f"{FNAME.stem}.{i:04d}.yaml"  # noqa
    with open(filename, "w") as f:
        yaml.dump(data, f)
