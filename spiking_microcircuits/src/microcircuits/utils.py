"""Utility classes for tracking and averaging during network simulation."""

import numpy as np
import numpy.typing as npt


class Tracker:
    """Records a target array over time, compressing samples by averaging.

    Records ``length * compress_len`` raw samples, compressed (averaged) into
    ``length`` stored entries in ``data``. The first entry in ``data`` is
    already an average over ``compress_len`` values, so the initial value of
    ``target`` does not generally equal ``data[0]`` when ``compress_len > 1``.
    Call ``finalize()`` after the simulation to flush any remaining buffered
    data into the last entry.

    Attributes:
        target: Reference to the array being tracked (updated in-place externally).
        data: Stored compressed samples, shape ``(length, *target.shape)``.
    """

    def __init__(self, length: int, target: npt.NDArray, compress_len: int) -> None:
        """Args:
        length: Number of compressed samples to store.
        target: Array to track (held by reference).
        compress_len: Number of raw samples averaged into one stored entry.
        """
        self.target = target
        self.data = np.zeros(tuple([length]) + target.shape, dtype=np.float32)
        self.index = 0
        self.buffer = np.zeros(target.shape)
        self.din = compress_len

    def record(self) -> None:
        """Accumulate the current target value; flush to data every compress_len calls."""
        self.buffer += self.target
        if (self.index + 1) % self.din == 0:
            self.data[int(self.index / self.din), :] = self.buffer / self.din
            self.buffer.fill(0)
        self.index += 1

    def finalize(self) -> None:
        """Flush any remaining buffered samples into the last data entry."""
        n_buffer = self.index % self.din
        if n_buffer > 0:
            self.data[int(self.index / self.din), :] = self.buffer / n_buffer


class SpikeTracker:
    """Tracks mean spike rates per neuron during simulation.

    Similar in concept to ``Tracker`` but records spikes as they occur rather
    than sampling a continuous array. Belongs to a layer rather than the
    network. Must be initialized via ``init_tracker`` before recording, because
    ``length`` and ``compress_len`` are not known at construction time.

    Attributes:
        mean_rates: Accumulated mean firing rates, shape ``(length, N_nrns)``.
    """

    def __init__(self, N_nrns: int, t_ref: float) -> None:
        """Args:
        N_nrns: Number of neurons to track.
        t_ref: Refractory period (used to convert spike counts to rates).
        """
        self.N_nrns = N_nrns
        self.t_ref = t_ref
        self.idx = 0  # index of the sample
        self.din = None
        self.mean_rates = None
        self.buffer = np.zeros(N_nrns)

    def init_tracker(self, length: int, compress_len: int) -> None:
        """Allocate storage. Call this before starting the simulation.

        Args:
            length: Number of compressed samples to store.
            compress_len: Number of time steps per stored sample.
        """
        self.index = 0
        self.din = compress_len
        self.mean_rates = np.zeros((length, self.N_nrns), dtype=np.float32)

    def record(self, nrn_id: int, t: float) -> None:
        """Register a spike for neuron ``nrn_id`` at time ``t``.

        Args:
            nrn_id: Index of the spiking neuron.
            t: Current simulation time step.
        """
        idx = int(t // self.din)  # find out to which "sample" the spike belongs
        self.mean_rates[idx, nrn_id] += 1.0

    def finalize(self, t_last: float) -> npt.NDArray:
        """Convert spike counts to rates and return the result.

        Args:
            t_last: Last simulation time step (used to handle partial final sample).

        Returns:
            Mean firing rates array of shape ``(length, N_nrns)``.
        """
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


class MovingAverage:
    """Online moving average over a fixed-size window.

    Maintains a running average of a 1-D array using a circular buffer of size
    ``stacksize``. During the initial fill phase (fewer than ``stacksize``
    updates), uses a cumulative average instead.

    Attributes:
        val: Current moving-average estimate.
    """

    def __init__(self, val: npt.NDArray, stacksize: int) -> None:
        """Args:
        val: Initial value; also determines the array shape and dtype.
        stacksize: Number of past values to average over.
        """
        self.val = val.copy()
        self.stack = np.zeros((stacksize, len(val)), dtype=val.dtype)
        self.stack[0] = self.val[:]
        self.stacksize = stacksize
        self.num_elements = 1
        self.i = 1  # stack index of element to be changed in next time step

    def move(self, new_val: npt.NDArray) -> None:
        """Update the moving average with a new observation.

        Args:
            new_val: New array value to incorporate.
        """
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
