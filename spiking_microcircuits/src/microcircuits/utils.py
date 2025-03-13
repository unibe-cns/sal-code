"""
Collection of miscellaneous function
"""

import numpy as np


def logistic(x, t_ref):
    return 1.0 / (1.0 + np.exp(-(x - np.log(t_ref))))


def rect_psp(dt, tau_syn):
    return np.heaviside(dt, 1.0) * np.heaviside(-(dt - tau_syn), 0.0)


def exp_stdp_kernel(dt, a, tau):
    return a * np.heaviside(dt, 0.0) * np.exp(-dt / tau)


def time_str(sec):
    """prints time to go string"""
    string = ""
    h = int(sec / 3600)
    if h > 0:
        string = str(h) + "h, "
    if int(sec / 60) > 0:
        m = int((sec - h * 3600) / 60)
        string += str(m) + "min and "
    string += str(int(sec % 60)) + "s"
    return string


def calc_spike_rates(spks, t_pattern, t_ref, num_pattern):
    spike_rate = []
    dt_pattern = t_pattern * t_ref
    len_sim = dt_pattern * num_pattern
    windows = np.arange(0, len_sim + 1, dt_pattern)
    for beg, end in zip(windows[:-1], windows[1:]):
        spks_window = np.array(spks[(spks >= beg) & (spks < end)])
        spike_rate.append(len(spks_window) / t_pattern)
    return np.array(spike_rate)


# taken directly from Laura
class Tracker:
    """
    Tracks/records changes in 'target' array. Records 'length'*'compress_len'
    samples, compressed (averaged) into 'length' samples. The result is stored
    in 'data'. Note that the first value in the 'data' is already the average
    of multiple values of the target array. If 'compress_len' is not 1 the
    initial value of 'target' is therefore not equal to the first entry in 'data'.
    After recording call finalize to also add the remaining data in buffer to
    'data' (finish the last compression).
    """

    def __init__(self, length, target, compress_len):
        self.target = target
        self.data = np.zeros(tuple([length]) + target.shape, dtype=np.float32)
        self.index = 0
        self.buffer = np.zeros(target.shape)
        self.din = compress_len

    def record(self):
        self.buffer += self.target
        if (self.index + 1) % self.din == 0:
            self.data[int(self.index / self.din), :] = self.buffer / self.din
            self.buffer.fill(0)
        self.index += 1

    def finalize(self):
        """fill last data point with average of remaining target data in buffer."""
        n_buffer = self.index % self.din
        if n_buffer > 0:
            self.data[int(self.index / self.din), :] = self.buffer / n_buffer


class SpikeTracker:
    """
    Tracks spikes during the simulation. Uses a similar concept as the normal
    Tracker, i.e. records the mean firing rate of a neuron per compression
    interval.
    Unlike the other tracker, this one belongs to the layer.
    """

    def __init__(self, N_nrns, t_ref):
        self.N_nrns = N_nrns
        self.t_ref = t_ref
        self.idx = 0  # index of the sample
        self.din = None
        self.mean_rates = None
        self.buffer = np.zeros(N_nrns)

    def init_tracker(self, length, compress_len):
        """actually initialize the tracker!

        I do it like this because the other trackers are just initialized in
        Net.run, therefore when initializing this class, length and compress_len
        are not known yet!
        """
        self.index = 0
        self.din = compress_len
        self.mean_rates = np.zeros((length, self.N_nrns), dtype=np.float32)

    def record(self, nrn_id, t):
        idx = int(t // self.din)  # find out to which "sample" the spike belongs
        self.mean_rates[idx, nrn_id] += 1.0

    def finalize(self, t_last):
        # check if the last sample was full or not!
        if (t_last % self.din) == 0:
            #  convert spike rate of full samples to units of t_ref^-1
            self.mean_rates /= self.din / self.t_ref
        # special treatment of the last sample:
        else:
            # treat samples that that were recorded for the full compress_len
            self.mean_rates[:-1] /= self.din / self.t_ref
            self.mean_rates[-1] /= (t_last % self.din) / self.t_ref
        return self.mean_rates


class RunResult(dict):
    """Collection of tracked data during a network run

    Basically a subclass of a dictionary...
    (inspired by https://github.com/scipy/scipy/blob/v1.8.1/scipy/optimize/_optimize.py#L84-L140)  # noqa

    # TODO: Docstring
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

    def get_rec_quant(self, quantity: str, layer: int, neuron_ids="all"):
        """Return the recorded value of a quantity

        layer must an integer: 1 for the first layer, and so on.
        neuron_ids can be 'all', than all neurons in the layer are passed, or
        a method to index a numpy array (np.s_ or np.array([1, 2]) or an integer)
        """
        if "rec_quants" not in self.keys():
            raise KeyError("No contineous quantities recorded!")

        layer_data = self["rec_quants"][layer - 1]
        if quantity not in layer_data:
            raise KeyError(f"{quantity} was not recorded in layer {layer}!")

        if neuron_ids == "all":
            res = layer_data[quantity].data
        else:
            res = layer_data[quantity].data[:, neuron_ids]

        # make sure that all arrays have same dimensions
        # (would be one-dim if neuron_ids is an integer)
        if len(res.shape) == 1:
            res = np.expand_dims(res, axis=1)

        return res

    def get_spks(self, nrn_type: str, layer: int, neuron_ids="all"):
        """
        neuron_ids can be an integer or a slice
        layer: 0 for input layer, ...
        """
        assert nrn_type in ["pyr", "inn", "input"]

        if nrn_type == "input":
            if neuron_ids == "all":
                return self["input_spikes"]
            elif type(neuron_ids) == int:
                return self["input_spikes"][slice(neuron_ids, neuron_ids + 1)]
            else:
                return self["input_spikes"][neuron_ids]

        sel = nrn_type + "_spikes"
        if neuron_ids == "all":
            res = self[sel][layer - 1]
        elif type(neuron_ids) == int:
            # this to make sure that a list is returned!
            res = self[sel][layer - 1][slice(neuron_ids, neuron_ids + 1)]
        else:
            res = self[sel][layer - 1][neuron_ids]

        return res


class MovingAverage:
    """docstring for MovingAverage."""

    def __init__(self, val, stacksize: int):
        self.val = val.copy()
        self.stack = np.zeros((stacksize, len(val)), dtype=val.dtype)
        self.stack[0] = self.val[:]
        self.stacksize = stacksize
        self.num_elements = 1
        self.i = 1  # stack index of element to be changed in next time step

    def move(self, new_val):
        # stack fills up at the beginning
        if self.num_elements < self.stacksize:
            self.num_elements += 1
            self.val += (new_val - self.val) / self.num_elements
        # stack is filled:
        else:
            last_val = self.stack[self.i]
            self.val += (new_val - last_val) / self.num_elements
        # update stack:
        self.stack[self.i] = new_val[:]
        self.i = (self.i + 1) % self.stacksize
