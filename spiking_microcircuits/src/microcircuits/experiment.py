#!/usr/bin/env python3

from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import yaml

from microcircuits import model as model
from microcircuits.model import WeightsList


class Descriptor(dict):
    """Holds parameters. Items behave also like normal class attributes

    Basically a subclass of a dictionary...
    (inspired by https://github.com/scipy/scipy/blob/v1.8.1/scipy/optimize/_optimize.py#L84-L140)  # noqa
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return "\n".join(
                [k.rjust(m) + ": " + repr(v) for k, v in sorted(self.items())]
            )
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())


class PropertiesDescriptor(Descriptor):
    """Holds all properties of some kind

    inherits from a dict. Includes automatic sanitychecks!
    """

    # TODO fill all attributes!
    _REQUIRED_ATTR = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._sanity_check()

    def _sanity_check(self):
        for item in self._REQUIRED_ATTR.items():
            assert item[0] in self.keys(), f"Keywork {item[0]} is missing!"
            assert isinstance(
                self[item[0]], item[1]
            ), f"Item {item[0]} has type {type(item[0])} but should be {item[1]}!"

    def _required_attr(self):
        return self._REQUIRED_ATTR


class NetworkProperties(PropertiesDescriptor):
    """Holds all network properties (= network internal parameters)

    inherits from a dict. Includes automatic sanitychecks!
    """

    # TODO fill all attributes!
    _REQUIRED_ATTR = {"t_ref": int, "n_last_spks": int}


class SimulationProperties(PropertiesDescriptor):
    """Holds all network properties (= network internal parameters)

    inherits from a dict. Includes automatic sanitychecks!
    """

    # TODO fill all attributes!
    _REQUIRED_ATTR = {"t_pattern": int, "len_epoch": int}


class InitialParameterDescriptor(PropertiesDescriptor):
    """Holds the initial parameters (weights and biases)"""

    # _REQUIRED_ATTR = {"weights": list, "bias": list}
    # TODO introduce something like optional attributes
    _REQUIRED_ATTR = {"bias": list}


class InputDescriptor(PropertiesDescriptor):
    """Holds the input values for the network"""

    _REQUIRED_ATTR = {"training": list, "validation": list}


class ExperimentDescriptor(Descriptor):
    def __init__(self, description_file: str):
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

    def todict(self):
        res = {}
        for attr, vals in self.items():
            res[attr] = dict(vals)
        return res


def check_network_properties(network_properties: dict):
    """DOCSTRING

    should check that all required params are present and have the correct type!
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


def check_simulation_settings(simulation_settings: dict):
    """DOCSTRING

    should check that all required params are present and have the correct type!
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


# TODO: validation and so on!


def run_teacher(
    network_properties: dict,
    teacher_parameters: dict,
    simulation_settings: dict,
    u_inp: dict,
    plotname: Optional[str] = None,
) -> dict:
    """DOCSTING

    TODO
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
    """Append the indexes of the pattern in an odrdered manner in an epoch

    TODO
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
    """DOCSTRING

    TODO
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
    limits: Union[list, tuple],
    seed: int,
    dims: list[int, int, int],
    down_limits: Optional[Union[list, tuple]] = None,
) -> WeightsList:
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
    network_properties: dict,
    student_parameters: dict,
    simulation_settings: dict,
    teacher_data: dict,
) -> dict:
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

    # TODO: best way to set the weights???

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


def test_all():
    exp = ExperimentDescriptor("example_experiment.yaml")

    teacher_res, t_res = run_teacher(
        exp.network_properties,
        exp.teacher_initial_parameters,
        exp.teacher_simulation_settings,
        exp.u_input,
        "teacher.png",
    )

    print("~~~~~~~~~~~~~~~~~~~~~~~ STUDENT ~~~~~~~~~~~~~~~~~~~~~~~~~")
    res = run_student(
        exp.network_properties,
        exp.student_initial_parameters,
        exp.student_simulation_settings,
        teacher_res,
    )

    fig, ax = plt.subplots(9, 1, sharex=True)
    i = 0
    ax[i].set_title("u_in")
    ax[i].plot(res["recordings"][0]["u_in"][:, 0])
    # ax[0].plot(t_res["recordings"][0]["u_in"][:, 0])

    i += 1
    ax[i].set_title("u pyr 1")
    ax[i].plot(res["recordings"][1]["u_pyr"][:, 0])
    # ax[1].plot(t_res["recordings"][1]["u_pyr"][:, 0])

    i += 1
    ax[i].set_title("u api 1")
    ax[i].plot(res["recordings"][1]["v_api"][:, 0])
    # ax[2].plot(t_res["recordings"][1]["v_api"][:, 0])

    i += 1
    ax[i].set_title("u pyr 2 and u tgt")
    ax[i].plot(res["recordings"][2]["u_pyr"][:, 0], label="u_pyr")
    # ax[3].plot(t_res["recordings"][2]["u_pyr"][:, 0])
    ax[i].plot(res["recordings"][2]["u_tgt"][:, 0], label="u_tgt")

    i += 1
    ax[i].set_title("u pyr 2 and u tgt")
    ax[i].plot(res["recordings"][2]["u_pyr"][:, 0], label="u_pyr")
    # ax[3].plot(t_res["recordings"][2]["u_pyr"][:, 0])
    ax[i].plot(res["recordings"][2]["u_tgt"][:, 0], label="u_tgt")

    i += 1
    ax[i].set_title("error: u_tgt - u_pyr")
    ax[i].plot(
        res["recordings"][2]["u_tgt"][:, 0] - res["recordings"][2]["u_pyr"][:, 0]
    )

    i += 1
    ax[i].set_title("w up 1 and w up 2")
    ax[i].plot(res["recordings"][1]["w_up"][:, 0, 0], label="w_up 1")
    ax[i].plot(res["recordings"][2]["w_up"][:, 0, 0], label="w_up 2")

    i += 1
    ax[i].set_title("w down 1")
    ax[i].plot(res["recordings"][1]["w_down"][:, 0, 0], label="w_down 1")

    i += 1
    ax[i].set_title("w pi and w ip")
    ax[i].plot(res["recordings"][1]["w_pi"][:, 0, 0], label="w_pi 1")
    ax[i].plot(res["recordings"][1]["w_ip"][:, 0, 0], label="w_ip 1")

    ts = np.arange(len(res["recordings"][0]["u_in"]))

    for axis in ax:
        axis.grid()
        axis.legend(loc="right")
        y_lower, y_upper = axis.get_ylim()
        axis.fill_between(
            ts,
            y_lower,
            y_upper,
            where=res["recordings"][2]["validation"].flatten(),
            color="lightgray",
            alpha=0.5,
        )
        axis.fill_between(
            ts,
            y_lower,
            y_upper,
            where=res["recordings"][2]["symmetrization"].flatten(),
            color="orange",
            alpha=0.5,
        )

    plt.tight_layout()
    plt.show()


def test_teacher():
    network_properties = {
        "t_ref": 25,
        "n_last_spks": 1,
        "dims": [1, 1, 1],
        "tau_syn": 25,
        "lambda_api": 0.5,
        "lambda_nudge": 0.5,
        "learning_lag": 0,
        "size_moving_average": 1000,
        "stdp_a_causal": 1.0,
        "stdp_a_anticausal": 1.0,
        "stdp_tau_causal": 25.0,
        "stdp_tau_anticausal": 25.0,
        "lr": [{"up": 0.0, "down": 0.0}, {"up": 0.0}],
    }

    u_in = {
        "validation": [-1.5, -0.5, 0.5, 1.5],
        "training": [-2.0, -1.0, 0.0, 1.0, 2.0],
    }

    teacher_parameters = {
        "weights": [
            {"w_up": 1.0},
            {"w_up": 2.0},
        ],
        "bias": [-0.0, -0.0],
    }

    simulation_settings = {
        "t_pattern": 500,
        "len_epoch": 5,
        "len_validation": 4,
        "num_epochs": 1,
        "poisson_seed": 78927,
        "training_seed": 90234,
        "shuffle_training": False,
        "shuffle_validation": False,
        "recorded_quantities": [
            ["u_in"],
            ["u_pyr", "v_bas", "v_api", "u_inn"],
            ["u_pyr", "v_bas", "v_nudge", "u_tgt"],
        ],
    }

    res = run_teacher(network_properties, teacher_parameters, simulation_settings, u_in)

    with open("some_teacher_data.yaml", "w") as f:
        yaml.dump(res, f)


def test_student():
    network_properties = {
        "t_ref": 25,
        "n_last_spks": 1,
        "dims": [1, 1, 1],
        "tau_syn": 25,
        "lambda_api": 0.5,
        "lambda_nudge": 0.5,
        "learning_lag": 0,
        "size_moving_average": 1000,
        "stdp_a_causal": 1.0,
        "stdp_a_anticausal": 1.0,
        "stdp_tau_causal": 25.0,
        "stdp_tau_anticausal": 25.0,
        "lr": [{"up": 0.0, "down": 0.0}, {"up": 0.0}],
    }

    with open("some_teacher_data.yaml", "r") as f:
        teacher_data = yaml.safe_load(f)

    initial_parameters = {
        "weights": [
            {"w_up": 1.0},
            {"w_up": 2.0},
        ],
        "bias": [-0.0, -0.0],
    }

    simulation_settings = {
        "t_pattern": 500,
        "len_epoch": 5,
        "len_validation": 4,
        "num_epochs": 1,
        "poisson_seed": 78927,
        "training_seed": 90234,
        "shuffle_training": False,
        "shuffle_validation": False,
        "recorded_quantities": [
            ["u_in"],
            ["u_pyr", "v_bas", "v_api", "u_inn", "w_up", "w_down"],
            ["u_pyr", "v_bas", "v_nudge", "u_tgt", "w_up"],
        ],
    }

    res = run_student(
        network_properties, initial_parameters, simulation_settings, teacher_data
    )

    fig, ax = plt.subplots(6, 1, sharex=True)
    ax[0].set_title("u_in")
    ax[0].plot(res["recordings"][0]["u_in"])

    ax[1].set_title("u pyr 1")
    ax[1].plot(res["recordings"][1]["u_pyr"])

    ax[2].set_title("u api 1")
    ax[2].plot(res["recordings"][1]["v_api"])

    ax[3].set_title("u pyr 2 and u tgt")
    ax[3].plot(res["recordings"][2]["u_pyr"])
    ax[3].plot(res["recordings"][2]["u_tgt"])

    ax[4].set_title("error: u_tgt - u_pyr")
    ax[4].plot(res["recordings"][2]["u_tgt"] - res["recordings"][2]["u_pyr"])

    ax[5].set_title("w up 1 and w up 2")
    ax[5].plot(res["recordings"][1]["w_up"].flatten())
    ax[5].plot(res["recordings"][2]["w_up"].flatten())

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # test_teacher()
    # test_student()
    test_all()
