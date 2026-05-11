#!/usr/bin/env python3

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import yaml

from microcircuits import model as model
from microcircuits.model import WeightsList


class Descriptor(dict):
    """Parameter container supporting both dict and attribute access.

    Basically a subclass of a dictionary.
    (inspired by https://github.com/scipy/scipy/blob/v1.8.1/scipy/optimize/_optimize.py#L84-L140)  # noqa
    """

    def __getattr__(self, name: str) -> Any:
        """Return item ``name`` as an attribute.

        Args:
            name: Key to look up.

        Raises:
            AttributeError: If ``name`` is not in the dict.
        """
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self) -> str:
        """Return a right-justified key-value listing."""
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return "\n".join(
                [k.rjust(m) + ": " + repr(v) for k, v in sorted(self.items())]
            )
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self) -> list[str]:
        """Return the list of available keys."""
        return list(self.keys())


class PropertiesDescriptor(Descriptor):
    """Parameter dict with automatic sanity checks on required attributes."""

    _REQUIRED_ATTR = {}

    def __init__(self, **kwargs: Any) -> None:
        """Args:
        **kwargs: Key-value pairs passed directly to the underlying dict.
        """
        super().__init__(**kwargs)
        self._sanity_check()

    def _sanity_check(self) -> None:
        """Assert all required attributes are present and have the correct type."""
        for item in self._REQUIRED_ATTR.items():
            assert item[0] in self.keys(), f"Keywork {item[0]} is missing!"
            assert isinstance(
                self[item[0]], item[1]
            ), f"Item {item[0]} has type {type(item[0])} but should be {item[1]}!"

    def _required_attr(self) -> dict[str, type]:
        """Return the required-attribute specification dict."""
        return self._REQUIRED_ATTR


class NetworkProperties(PropertiesDescriptor):
    """Network internal parameters (requires ``t_ref`` and ``n_last_spks``)."""

    # TODO fill all attributes!
    _REQUIRED_ATTR = {"t_ref": int, "n_last_spks": int}


class SimulationProperties(PropertiesDescriptor):
    """Simulation settings (requires ``t_pattern`` and ``len_epoch``)."""

    # TODO fill all attributes!
    _REQUIRED_ATTR = {"t_pattern": int, "len_epoch": int}


class InitialParameterDescriptor(PropertiesDescriptor):
    """Initial weight and bias parameters (requires ``bias``)."""

    # _REQUIRED_ATTR = {"weights": list, "bias": list}
    # TODO introduce something like optional attributes
    _REQUIRED_ATTR = {"bias": list}


class InputDescriptor(PropertiesDescriptor):
    """Input data descriptor (requires ``training`` and ``validation`` lists)."""

    _REQUIRED_ATTR = {"training": list, "validation": list}


class ExperimentDescriptor(Descriptor):
    """Loads and validates all experiment settings from a YAML description file."""

    def __init__(self, description_file: str) -> None:
        """Args:
        description_file: Path to the YAML file describing the experiment.
        """
        with open(description_file, "r") as f:
            data = yaml.safe_load(f)

        self.network_properties = NetworkProperties(**data["network_properties"])

        if "teacher_simulation_settings" in data:
            self.teacher_simulation_settings = SimulationProperties(
                **data["teacher_simulation_settings"]
            )
        else:
            self.teacher_simulation_settings = None

        if "student_simulation_settings" in data:
            self.student_simulation_settings = SimulationProperties(
                **data["student_simulation_settings"]
            )
        else:
            self.student_simulation_settings = None

        if "teacher_initial_parameters" in data:
            self.teacher_initial_parameters = InitialParameterDescriptor(
                **data["teacher_initial_parameters"]
            )
        else:
            self.teacher_initial_parameters = None

        if "student_initial_parameters" in data:
            self.student_initial_parameters = InitialParameterDescriptor(
                **data["student_initial_parameters"]
            )
        else:
            self.student_initial_parameters = None

        self.u_input = InputDescriptor(**data["u_input"])

    def todict(self) -> dict[str, dict[str, Any]]:
        """Return a plain nested dict representation of all settings.

        Returns:
            Dict mapping each descriptor attribute name to its contents as a dict.
        """
        res = {}
        for attr, vals in self.items():
            res[attr] = dict(vals)
        return res


def check_network_properties(network_properties: dict[str, Any]) -> None:
    """Validate that all required network parameter keys are present and correctly typed.

    Args:
        network_properties: Dict of network parameters to validate.

    Raises:
        AssertionError: If a required key is missing or has the wrong type.
    """
    assert "t_ref" in network_properties
    assert isinstance(network_properties["t_ref"], int)
    assert "n_last_spks" in network_properties
    assert isinstance(network_properties["n_last_spks"], int)
    assert "dims" in network_properties
    assert isinstance(network_properties["dims"], list)
    assert "tau_syn" in network_properties
    assert isinstance(network_properties["tau_syn"], int)
    assert "lambda_api" in network_properties
    assert isinstance(network_properties["lambda_api"], float)
    assert "lambda_nudge" in network_properties
    assert isinstance(network_properties["lambda_nudge"], float)
    assert "learning_lag" in network_properties
    assert isinstance(network_properties["learning_lag"], int)
    assert "size_moving_average" in network_properties
    assert isinstance(network_properties["size_moving_average"], int)
    assert "stdp_a_causal" in network_properties
    assert isinstance(network_properties["stdp_a_causal"], float)
    assert "stdp_a_anticausal" in network_properties
    assert isinstance(network_properties["stdp_a_anticausal"], float)
    assert "stdp_tau_causal" in network_properties
    assert isinstance(network_properties["stdp_tau_causal"], float)
    assert "stdp_tau_anticausal" in network_properties
    assert isinstance(network_properties["stdp_tau_anticausal"], float)
    assert "lr" in network_properties
    assert isinstance(network_properties["lr"], list)


def check_simulation_settings(simulation_settings: dict[str, Any]) -> None:
    """Validate that all required simulation setting keys are present and correctly typed.

    Args:
        simulation_settings: Dict of simulation settings to validate.

    Raises:
        AssertionError: If a required key is missing or has the wrong type.
    """
    assert "poisson_seed" in simulation_settings
    assert isinstance(simulation_settings["poisson_seed"], int)
    assert "training_seed" in simulation_settings
    assert isinstance(simulation_settings["training_seed"], int)
    assert "t_pattern" in simulation_settings
    assert isinstance(simulation_settings["t_pattern"], int)
    assert "recorded_quantities" in simulation_settings
    assert isinstance(simulation_settings["recorded_quantities"], list)
    assert "len_epoch" in simulation_settings
    assert isinstance(simulation_settings["len_epoch"], int)
    assert "len_validation" in simulation_settings
    assert isinstance(simulation_settings["len_validation"], int)
    assert "num_epochs" in simulation_settings
    assert isinstance(simulation_settings["num_epochs"], int)
    assert "shuffle_training" in simulation_settings
    assert isinstance(simulation_settings["shuffle_training"], int)
    assert "shuffle_validation" in simulation_settings
    assert isinstance(simulation_settings["shuffle_validation"], int)


def run_teacher(
    network_properties: dict[str, Any],
    teacher_parameters: dict[str, Any],
    simulation_settings: dict[str, Any],
    u_inp: dict[str, Any],
    plotname: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the teacher network and return input-target pairs and raw simulation results.

    Args:
        network_properties: Validated network parameter dict.
        teacher_parameters: Dict with ``"weights"`` (or ``"random_weights_init_limits"``)
            and ``"bias"`` keys.
        simulation_settings: Validated simulation settings dict.
        u_inp: Dict with ``"training"`` and ``"validation"`` input voltage lists.
        plotname: If provided, save an input-vs-target scatter plot to this path.

    Returns:
        Tuple of ``(teacher_data, raw_results)`` where ``teacher_data`` contains
        ``"u_inp"`` and ``"u_target"`` dicts for training and validation sets.
    """

    print("~~~~~~~~~~~~~~~~~~~~~~~ TEACHER ~~~~~~~~~~~~~~~~~~~~~~~~~")
    check_network_properties(network_properties)
    check_simulation_settings(simulation_settings)

    net = model.Network(network_properties, simulation_settings["poisson_seed"])

    if "weights" in teacher_parameters:
        net.set_weights(teacher_parameters["weights"])
    elif "random_weights_init_limits" in teacher_parameters:
        assert "weights_init_seed" in simulation_settings
        random_weights = draw_random_weights(
            teacher_parameters["random_weights_init_limits"],
            simulation_settings["weights_init_seed"],
            network_properties["dims"],
        )
        net.set_weights(random_weights)
        print("random init weights:")
        print(random_weights)
    else:
        raise KeyError(
            "You need either the key 'weights' or 'random_weights_init_limits'!"
        )

    net.distribute_weights()

    net.set_bias(teacher_parameters["bias"])

    # how many time steps are compressed to one recoring sample
    compress_len = network_properties["t_ref"] * simulation_settings["t_pattern"]

    total_inp = np.concatenate((u_inp["training"], u_inp["validation"]))

    res = net.run(
        total_inp,
        simulation_settings["t_pattern"],
        simulation_settings["recorded_quantities"],
        compress_len,
        len_epoch=len(u_inp["training"]),
        validation_len=len(u_inp["validation"]),
        update_up=False,
        update_down=False,
        record_all_spks=False,
    )

    u_target = {
        "training": res["u_out"][: len(u_inp["training"]), 0].tolist(),
        "validation": res["u_out"][len(u_inp["training"]) :, 0].tolist(),
    }

    if plotname is not None:
        fig, ax = plt.subplots()
        ax.plot(u_inp["training"], u_target["training"], "o", label="training")
        ax.plot(u_inp["validation"], u_target["validation"], "o", label="validation")
        ax.set_xlabel("u input")
        ax.set_ylabel("u target")
        plt.tight_layout()
        fig.savefig(plotname)

    return {"u_inp": u_inp, "u_target": u_target}, res


def prepare_ordered_epoch(epoch_ids: np.ndarray, len_epoch: int) -> npt.NDArray:
    """Repeat and trim pattern indices to exactly ``len_epoch`` entries.

    Args:
        epoch_ids: Array of pattern indices to repeat.
        len_epoch: Target length of the returned index array.

    Returns:
        Index array of length ``len_epoch`` with ``epoch_ids`` tiled and trimmed.
    """
    num_full_repeats = int(np.ceil(len_epoch / len(epoch_ids)))
    if num_full_repeats > 0:
        return np.concatenate((epoch_ids,) * num_full_repeats)[:len_epoch]
    else:
        return np.array([], dtype=int)


def shuffle_training_data(
    x_train: npt.ArrayLike,
    x_val: npt.ArrayLike,
    y_train: npt.ArrayLike,
    y_val: npt.ArrayLike,
    seed: int,
    len_epoch: int,
    len_validation: int,
    num_epochs: int,
    shuffle_val: bool = True,
    shuffle_training: bool = True,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Build the full training sequence by interleaving shuffled training and validation data.

    For each epoch, ``len_epoch`` training samples and ``len_validation`` validation
    samples are drawn (with replacement) and concatenated. A ``val_mask`` indicates
    which patterns belong to training (1) and which to validation (0).

    Args:
        x_train: Training inputs.
        x_val: Validation inputs.
        y_train: Training targets.
        y_val: Validation targets.
        seed: Seed for the random number generator.
        len_epoch: Number of training samples per epoch.
        len_validation: Number of validation samples per epoch.
        num_epochs: Number of training epochs.
        shuffle_val: Whether to draw validation samples randomly (else ordered).
        shuffle_training: Whether to draw training samples randomly (else ordered).

    Returns:
        Tuple ``(all_x, all_y, val_mask)`` where ``val_mask`` is 1 for training
        patterns and 0 for validation patterns.
    """
    rng = np.random.default_rng(seed)

    expand = lambda x: x[:, np.newaxis] if x.ndim == 1 else x

    x_train = expand(np.array(x_train))
    x_val = expand(np.array(x_val))
    y_train = expand(np.array(y_train))
    y_val = expand(np.array(y_val))

    ## prepare the data:
    epoch_ids = np.arange(x_train.shape[0])  # needed for shuffling the data
    val_ids = np.arange(x_val.shape[0])

    all_x = np.empty((0, x_train.shape[1]))
    all_y = np.empty((0, y_train.shape[1]))
    val_mask = np.empty(0)

    for i in range(num_epochs):
        # append training data:
        if shuffle_training:
            shuffled_train_idx = rng.choice(epoch_ids, len_epoch)
        else:
            shuffled_train_idx = prepare_ordered_epoch(epoch_ids, len_epoch)
        all_x = np.vstack((all_x, x_train[shuffled_train_idx]))
        # y_train_shuffled = np.hstack(
        #     (y_train[shuffled_train_idx], np.ones((len_epoch, 1)))
        # )
        all_y = np.vstack((all_y, y_train[shuffled_train_idx]))
        val_mask = np.append(val_mask, np.ones(len_epoch))

        # append validation data:
        if shuffle_val:
            shuffled_val_idx = rng.choice(val_ids, len_validation)
        else:
            shuffled_val_idx = prepare_ordered_epoch(val_ids, len_validation)
        all_x = np.vstack((all_x, x_val[shuffled_val_idx]))
        # y_val_shuffled = np.hstack(
        #     (y_val[shuffled_val_idx], np.zeros((len_validation, 1)))
        # )
        all_y = np.vstack((all_y, y_val[shuffled_val_idx]))
        val_mask = np.append(val_mask, np.zeros(len_validation))

    return all_x, all_y, val_mask


def draw_random_weights(
    limits: tuple[float, float] | list[float],
    seed: int,
    dims: list[int],
    down_limits: tuple[float, float] | list[float] | None = None,
) -> WeightsList:
    """Draw uniform random weights for a 3-layer network.

    Args:
        limits: ``(low, high)`` range for upward weights.
        seed: Seed for the random number generator.
        dims: Layer dimensions ``[n_in, n_hidden, n_out]``.
        down_limits: ``(low, high)`` range for downward weights; defaults to ``limits``.

    Returns:
        ``WeightsList`` with one dict per non-input layer containing ``"w_up"``
        and (for hidden layers) ``"w_down"`` arrays.
    """
    down_limits = limits if down_limits is None else down_limits
    rng = np.random.default_rng(seed)
    weights = [
        {
            "w_up": rng.uniform(limits[0], limits[1], size=(dims[1], dims[0])),
            "w_down": rng.uniform(
                down_limits[0], down_limits[1], size=(dims[1], dims[2])
            ),
        },
        {"w_up": rng.uniform(limits[0], limits[1], size=(dims[2], dims[1]))},
    ]

    return weights


def run_student(
    network_properties: dict[str, Any],
    student_parameters: dict[str, Any],
    simulation_settings: dict[str, Any],
    teacher_data: dict[str, Any],
) -> dict[str, Any]:
    """Train the student network and return simulation results.

    Args:
        network_properties: Validated network parameter dict.
        student_parameters: Dict with ``"weights"`` (or ``"random_weights_init_limits"``)
            and ``"bias"`` keys.
        simulation_settings: Validated simulation settings dict.
        teacher_data: Dict with ``"u_inp"`` and ``"u_target"`` as produced by
            :func:`run_teacher`.

    Returns:
        Raw simulation result dict from :meth:`~microcircuits.model.Network.run`,
        with an additional ``"random_weights_init"`` entry.
    """
    print("~~~~~~~~~~~~~~~~~~~~~~~ STUDENT ~~~~~~~~~~~~~~~~~~~~~~~~~")

    check_network_properties(network_properties)
    check_simulation_settings(simulation_settings)

    net = model.Network(network_properties, simulation_settings["poisson_seed"])

    if "weights" in student_parameters:
        net.set_weights(student_parameters["weights"])

    elif "random_weights_init_limits" in student_parameters:
        assert "weights_init_seed" in simulation_settings
        random_weights = draw_random_weights(
            student_parameters["random_weights_init_limits"]["up"],
            simulation_settings["weights_init_seed"],
            network_properties["dims"],
            down_limits=student_parameters["random_weights_init_limits"]["down"],
        )
        net.set_weights(random_weights)
        print("random init weights:")
        print(random_weights)

    else:
        raise KeyError(
            "You need either the key 'weights' or 'random_weights_init_limits'!"
        )

    if simulation_settings["set_sps"]:
        net.distribute_weights(update_down=simulation_settings["update_down"])

    net.set_bias(student_parameters["bias"])

    # how many time steps are compressed to one recoring sample

    u_in, u_tgt, val_mask = shuffle_training_data(
        teacher_data["u_inp"]["training"],
        teacher_data["u_inp"]["validation"],
        teacher_data["u_target"]["training"],
        teacher_data["u_target"]["validation"],
        simulation_settings["training_seed"],
        simulation_settings["len_epoch"],
        simulation_settings["len_validation"],
        simulation_settings["num_epochs"],
        shuffle_training=simulation_settings["shuffle_training"],
        shuffle_val=simulation_settings["shuffle_validation"],
    )

    res = net.run(
        u_in,
        simulation_settings["t_pattern"],
        simulation_settings["recorded_quantities"],
        simulation_settings["recorded_sample_length"],
        simulation_settings["len_epoch"],
        simulation_settings["len_validation"],
        u_tgt=u_tgt,
        update_up=True,
        update_down=simulation_settings["update_down"],
        set_sps=simulation_settings["set_sps"],
        record_all_spks=simulation_settings["record_all_spks"],
        len_symm=simulation_settings["len_symmetrization"],
    )

    res["random_weights_init"] = random_weights
    return res
