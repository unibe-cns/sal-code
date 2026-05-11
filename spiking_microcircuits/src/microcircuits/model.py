#!/usr/bin/env python3

"""Spiking microcircuit model: InputLayer, Layer, OutputLayer, and Network."""

import time
from itertools import repeat
from typing import Any

import numpy as np
import numpy.typing as npt

from .utils import MovingAverage, Tracker

AllSpikes = list[list[float]]
WeightsList = list[dict[str, npt.ArrayLike]]


def rect_psp(dt: npt.NDArray, tau_syn: float) -> npt.NDArray:
    """Rectangular post-synaptic potential kernel.

    Returns 1 if ``0 <= dt < tau_syn``, 0 otherwise.

    Args:
        dt: Time since pre-synaptic spike.
        tau_syn: Synaptic time constant (width of the rectangular window).

    Returns:
        Binary array with 1 inside the window and 0 outside.
    """
    return np.heaviside(dt, 1.0) * np.heaviside(-(dt - tau_syn), 0.0)


def logistic(u: npt.NDArray, t_ref: float) -> npt.NDArray:
    """Logistic activation function mapping membrane potential to firing probability.

    Args:
        u: Membrane potential array.
        t_ref: Refractory period, sets the threshold via ``log(t_ref)``.

    Returns:
        Firing probability array, same shape as ``u``.
    """
    return 1.0 / (1.0 + np.exp(-(u - np.log(t_ref))))


def exp_stdp_kernel(dt: npt.NDArray, a: float, tau: float) -> npt.NDArray:
    """Exponential STDP kernel.

    Returns ``a * exp(-dt / tau)`` for ``dt >= 0``, 0 otherwise.

    Args:
        dt: Spike timing difference (post minus pre).
        a: Amplitude of the kernel.
        tau: Decay time constant.

    Returns:
        STDP kernel values, same shape as ``dt``.
    """
    return a * np.heaviside(dt, 0.0) * np.exp(-dt / tau)


def loss_func(out: npt.NDArray, tgt: npt.NDArray | float) -> float:
    """Mean squared error loss.

    Args:
        out: Network output array.
        tgt: Target array or scalar.

    Returns:
        Scalar MSE loss.
    """
    return np.sum((out - tgt) ** 2) / len(out)


class InputLayer:
    """Virtual input layer that converts an input voltage into Poisson spike trains."""

    def __init__(
        self,
        dims: int,
        params: dict[str, Any],
        rng: np.random.Generator,
    ) -> None:
        """Args:
        dims: Number of input neurons.
        params: Network parameter dict (requires ``n_last_spks``, ``size_moving_average``, ``t_ref``).
        rng: Random number generator for spike sampling.
        """
        self.dims = dims
        self.last_spks = np.full((dims, params["n_last_spks"]), -np.inf)
        self.all_spks: list[list[float]] = [[] for i in range(dims)]
        self.mean_rates_pattern: list[float] = []
        self.mean_rates_buffer = np.zeros(dims)
        self.u_in = np.zeros(dims)
        self.u_in_av = MovingAverage(self.u_in, params["size_moving_average"])
        self.rng = rng
        self.record_all_spks = True

        self.set_params(params)

    def set_params(self, params: dict[str, Any]) -> None:
        """Load ``t_ref`` and ``n_last_spks`` from the parameter dict.

        Args:
            params: Network parameter dict.
        """
        self.t_ref = params["t_ref"]
        self.n_last_spks = params["n_last_spks"]

    def update_mempot(self, u_in: npt.NDArray) -> None:
        """Set the input voltage and update its moving average.

        Args:
            u_in: New input voltage array.
        """
        self.u_in[:] = u_in
        self.u_in_av.move(self.u_in)

    def update_spks(self, t: float) -> None:
        """Sample spikes stochastically from the logistic activation at time ``t``.

        Args:
            t: Current simulation time step.
        """
        self.r_in = logistic(self.u_in, self.t_ref)
        random_vals = self.rng.random(self.dims)
        for i in np.nonzero(random_vals < self.r_in)[0]:
            if self.last_spks[i, -1] <= t - self.t_ref:
                if self.record_all_spks:
                    self.all_spks[i].append(t)
                self.last_spks[i, :-1] = self.last_spks[i, 1:]
                self.last_spks[i, -1] = t
                self.mean_rates_buffer[i] += 1.0

    def calc_rates_pattern(self, t_pattern: float) -> None:
        """Append mean spike rates for the current pattern and reset the buffer.

        Args:
            t_pattern: Duration of the current pattern in time steps.
        """
        self.mean_rates_pattern.append(self.mean_rates_buffer / t_pattern)
        self.mean_rates_buffer = np.zeros(self.dims)

    def get_all_spks(self) -> dict[str, list[list[float]]]:
        """Return all recorded spike times.

        Returns:
            Dict with key ``"inp"`` mapping to a list of per-neuron spike-time lists.
        """
        return {"inp": self.all_spks}


class Layer:
    """Hidden cortical microcircuit layer with pyramidal neurons and interneurons."""

    def __init__(
        self,
        n_pyr: int,
        n_in: int,
        n_next: int,
        params: dict[str, Any],
        layer_id: int,
        rng: np.random.Generator,
    ) -> None:
        """Args:
        n_pyr: Number of pyramidal neurons.
        n_in: Number of neurons in the previous (input-side) layer.
        n_next: Number of neurons in the next layer.
        params: Network parameter dict.
        layer_id: 1-based index of this layer (used to index learning rates).
        rng: Random number generator for spike sampling.
        """
        self.N_PYR = n_pyr
        self.N_IN = n_in
        self.N_NEXT = n_next
        self.LAYER_ID = layer_id

        self.u_pyr = np.zeros(self.N_PYR)
        self.v_bas = np.zeros_like(self.u_pyr)
        self.v_api = np.zeros_like(self.u_pyr)

        self.u_pyr_av = MovingAverage(self.u_pyr, params["size_moving_average"])
        self.v_bas_av = MovingAverage(self.v_bas, params["size_moving_average"])

        self.u_inn = np.zeros(self.N_NEXT)

        self.record_all_spks = True
        self.last_spks_pyr = np.full((self.N_PYR, params["n_last_spks"]), -np.inf)
        self.all_spks_pyr: list[list[float]] = [[] for i in range(self.N_PYR)]
        self.rates_pattern_pyr: list[npt.NDArray] = []
        self.rates_buffer_pyr = np.zeros(self.N_PYR)

        self.last_spks_inn = np.full((self.N_NEXT, params["n_last_spks"]), -np.inf)
        self.all_spks_inn: list[list[float]] = [[] for i in range(self.N_NEXT)]
        self.rates_pattern_inn: list[npt.NDArray] = []
        self.rates_buffer_inn = np.zeros(self.N_NEXT)

        self.rng = rng

        self.w_up = np.zeros((self.N_PYR, self.N_IN))
        self.w_down = np.zeros((self.N_PYR, self.N_NEXT))
        self.w_pi = np.zeros((self.N_PYR, self.N_NEXT))
        self.w_ip = np.zeros((self.N_NEXT, self.N_PYR))
        self.stdp = np.zeros_like(self.w_up)

        self.b_pyr = np.zeros_like(self.u_pyr)
        self.b_inn = np.zeros_like(self.u_inn)

        self.set_params(params)

    def set_params(self, params: dict[str, Any]) -> None:
        """Load synaptic and plasticity parameters from the parameter dict.

        Args:
            params: Network parameter dict.
        """
        self.n_last_spks = params["n_last_spks"]
        self.t_ref = params["t_ref"]
        self.tau_syn = params["tau_syn"]
        self.lam_api = params["lambda_api"]
        self.stdp_a_causal = params["stdp_a_causal"]
        self.stdp_a_anticausal = params["stdp_a_anticausal"]
        self.stdp_tau_causal = params["stdp_tau_causal"]
        self.stdp_tau_anticausal = params["stdp_tau_anticausal"]
        self.lr_up = params["lr"][self.LAYER_ID - 1]["up"]
        self.lr_down = params["lr"][self.LAYER_ID - 1]["down"]

    def set_bias(self, bias: npt.ArrayLike, bias_next: npt.ArrayLike) -> None:
        """Set biases for the pyramidal and interneuron populations.

        Args:
            bias: Bias vector for pyramidal neurons.
            bias_next: Bias vector for interneurons.
        """
        np.copyto(dst=self.b_pyr, src=bias)
        np.copyto(dst=self.b_inn, src=bias_next)

    def set_weights_random(self):
        raise NotImplementedError

    def set_weights(
        self,
        w_up: npt.ArrayLike | None = None,
        w_down: npt.ArrayLike | None = None,
        w_pi: npt.ArrayLike | None = None,
        w_ip: npt.ArrayLike | None = None,
    ) -> None:
        """Copy provided weight matrices in-place; ``None`` arguments are skipped.

        Args:
            w_up: Bottom-up weights, shape ``(N_PYR, N_IN)``.
            w_down: Top-down feedback weights, shape ``(N_PYR, N_NEXT)``.
            w_pi: Pyramidal-to-interneuron weights, shape ``(N_PYR, N_NEXT)``.
            w_ip: Interneuron-to-pyramidal weights, shape ``(N_NEXT, N_PYR)``.
        """
        if w_up is not None:
            np.copyto(src=w_up, dst=self.w_up)
        if w_down is not None:
            np.copyto(src=w_down, dst=self.w_down)
        if w_pi is not None:
            np.copyto(src=w_pi, dst=self.w_pi)
        if w_ip is not None:
            np.copyto(src=w_ip, dst=self.w_ip)

    def get_weights(self) -> dict[str, npt.NDArray]:
        """Return copies of all weight matrices.

        Returns:
            Dict with keys ``"w_up"``, ``"w_down"``, ``"w_ip"``, ``"w_pi"``.
        """
        return {
            "w_up": self.w_up.copy(),
            "w_down": self.w_down.copy(),
            "w_ip": self.w_ip.copy(),
            "w_pi": self.w_pi.copy(),
        }

    def set_sps(self, w_up_next: npt.NDArray) -> None:
        """Initialize symmetric plasticity schedule (SPS) weights from the next layer's ``w_up``.

        Sets ``w_ip = w_up_next`` and ``w_pi = -w_down``.

        Args:
            w_up_next: Bottom-up weight matrix of the next layer.
        """
        self.w_ip[:, :] = w_up_next
        self.w_pi[:, :] = -self.w_down

    def update_pyr_spks(self, t: float) -> None:
        """Sample pyramidal neuron spikes stochastically at time ``t``.

        Args:
            t: Current simulation time step.
        """
        r_pyr = logistic(self.u_pyr, self.t_ref)
        random_vals = self.rng.random(self.N_PYR)
        for i in np.nonzero(random_vals < r_pyr)[0]:
            if self.last_spks_pyr[i, -1] <= t - self.t_ref:
                if self.record_all_spks:
                    self.all_spks_pyr[i].append(t)
                self.last_spks_pyr[i, :-1] = self.last_spks_pyr[i, 1:]
                self.last_spks_pyr[i, -1] = t
                self.rates_buffer_pyr[i] += 1.0

    def update_inn_spks(self, t: float) -> None:
        """Sample interneuron spikes stochastically at time ``t``.

        Args:
            t: Current simulation time step.
        """
        r_inn = logistic(self.u_inn, self.t_ref)
        random_vals = self.rng.random(self.N_NEXT)
        for i in np.nonzero(random_vals < r_inn)[0]:
            if self.last_spks_inn[i, -1] <= t - self.t_ref:
                if self.record_all_spks:
                    self.all_spks_inn[i].append(t)
                self.last_spks_inn[i, :-1] = self.last_spks_inn[i, 1:]
                self.last_spks_inn[i, -1] = t
                self.rates_buffer_inn[i] += 1.0

    def update_spks(self, t: float) -> None:
        """Update spikes for both pyramidal and interneuron populations.

        Args:
            t: Current simulation time step.
        """
        self.update_pyr_spks(t)
        self.update_inn_spks(t)

    def calc_rates_pattern(self, t_pattern: float) -> None:
        """Append mean rates for the current pattern and reset spike buffers.

        Args:
            t_pattern: Duration of the current pattern in time steps.
        """
        self.rates_pattern_pyr.append(self.rates_buffer_pyr / t_pattern)
        self.rates_buffer_pyr.fill(0.0)
        self.rates_pattern_inn.append(self.rates_buffer_inn / t_pattern)
        self.rates_buffer_inn.fill(0.0)

    def dendritic_voltages(
        self, t: float, last_spks: npt.NDArray, weight: npt.NDArray
    ) -> npt.NDArray:
        """Compute weighted sum of rectangular PSPs from a spike history.

        Args:
            t: Current simulation time step.
            last_spks: Recent spike times, shape ``(n_pre, n_last_spks)``.
            weight: Synaptic weight matrix, shape ``(n_post, n_pre)``.

        Returns:
            Dendritic voltage contribution, shape ``(n_post,)``.
        """
        psps = np.sum(rect_psp(t - last_spks, self.tau_syn), axis=1)
        return np.dot(weight, psps)

    def update_pyr_mempot(
        self,
        t: float,
        last_spks_in: npt.NDArray,
        last_spks_next: npt.NDArray,
    ) -> None:
        """Update pyramidal membrane potential from basal and apical compartments.

        Args:
            t: Current simulation time step.
            last_spks_in: Recent spike times of the previous layer.
            last_spks_next: Recent spike times of the next layer's pyramidal neurons.
        """
        self.v_bas[:] = self.dendritic_voltages(t, last_spks_in, self.w_up)
        self.v_api[:] = self.dendritic_voltages(
            t, last_spks_next, self.w_down
        ) + self.dendritic_voltages(t, self.last_spks_inn, self.w_pi)

        self.u_pyr[:] = self.b_pyr + self.v_bas + self.lam_api * self.v_api

        self.u_pyr_av.move(self.u_pyr)
        self.v_bas_av.move(self.v_bas)

    def update_inn_mempot(self, t: float) -> None:
        """Update interneuron membrane potential from pyramidal spikes.

        Args:
            t: Current simulation time step.
        """
        self.u_inn[:] = self.b_inn + self.dendritic_voltages(
            t, self.last_spks_pyr, self.w_ip
        )

    def update_mempot(
        self, t: float, last_spks_in: npt.NDArray, last_spks_next: npt.NDArray
    ) -> None:
        """Update membrane potentials for pyramidal and interneuron populations.

        Args:
            t: Current simulation time step.
            last_spks_in: Recent spike times of the previous layer.
            last_spks_next: Recent spike times of the next layer's pyramidal neurons.
        """
        self.update_pyr_mempot(t, last_spks_in, last_spks_next)
        self.update_inn_mempot(t)

    def plasticity_up(self, t: float, last_spks_in: npt.NDArray) -> None:
        """Apply RDD-based upward weight update.

        Args:
            t: Current simulation time step.
            last_spks_in: Recent spike times of the previous layer.
        """
        now_spks_in = np.where(last_spks_in[:, -1] == t - 1, 1, 0)
        r_bas_hat = logistic(self.v_bas_av.val + self.b_pyr, self.t_ref)
        r_pyr_hat = logistic(self.u_pyr_av.val, self.t_ref)
        delta_up = np.outer(r_pyr_hat - r_bas_hat, now_spks_in)
        self.w_up[:, :] += self.lr_up * delta_up

    def plasticity_down(self, t: float, last_spks_next: npt.NDArray) -> None:
        """Apply STDP-based downward weight update.

        Args:
            t: Current simulation time step.
            last_spks_next: Recent spike times of the next layer's pyramidal neurons.
        """
        self.stdp.fill(0.0)

        post_spk_idx = np.array(np.nonzero(self.last_spks_pyr == t))
        if post_spk_idx.shape[1] > 0:
            dts_pre = t - last_spks_next
            traces_pre = np.sum(
                exp_stdp_kernel(dts_pre, self.stdp_a_causal, self.stdp_tau_causal),
                axis=1,
            )
            traces_pre /= self.n_last_spks
            self.stdp[post_spk_idx[0]] = traces_pre

        pre_spks_idx = np.array(np.nonzero(last_spks_next == t))
        if pre_spks_idx.shape[1] > 0:
            dts_post = t - self.last_spks_pyr
            traces_post = np.sum(
                exp_stdp_kernel(
                    dts_post, self.stdp_a_anticausal, self.stdp_tau_anticausal
                ),
                axis=1,
            )
            traces_post /= self.n_last_spks
            self.stdp[:, pre_spks_idx[0]] = traces_post[:, np.newaxis]

        self.w_down[:, :] += self.lr_down * self.stdp

    def copybackprob(self, w_up_next: npt.NDArray) -> None:
        """Set ``w_down`` to the transpose of the next layer's ``w_up`` (copy-back propagation).

        Args:
            w_up_next: Bottom-up weight matrix of the next layer.
        """
        self.w_down[:, :] = w_up_next.T

    def get_all_spks(self) -> dict[str, list[list[float]]]:
        """Return all recorded spike times for both populations.

        Returns:
            Dict with keys ``"pyr"`` and ``"inn"``, each a list of per-neuron spike-time lists.
        """
        return {"pyr": self.all_spks_pyr, "inn": self.all_spks_inn}


class OutputLayer:
    """Output cortical microcircuit layer with nudging via a target signal."""

    def __init__(
        self,
        n_out: int,
        n_in: int,
        params: dict[str, Any],
        layer_id: int,
        rng: np.random.Generator,
    ) -> None:
        """Args:
        n_out: Number of output neurons.
        n_in: Number of neurons in the previous layer.
        params: Network parameter dict.
        layer_id: 1-based index of this layer (used to index learning rates).
        rng: Random number generator for spike sampling.
        """
        self.N_OUT = n_out
        self.N_IN = n_in
        self.LAYER_ID = layer_id

        self.u_pyr = np.zeros(self.N_OUT)
        self.v_bas = np.zeros_like(self.u_pyr)
        self.v_bas_av = MovingAverage(self.v_bas, params["size_moving_average"])
        self.u_pyr_av = MovingAverage(self.u_pyr, params["size_moving_average"])

        self.v_nudge = np.zeros_like(self.u_pyr)
        self.u_tgt = np.zeros_like(self.u_pyr)
        self.u_tgt_av = MovingAverage(self.u_tgt, params["size_moving_average"])

        self.record_all_spks = True
        self.last_spks_pyr = np.full((self.N_OUT, params["n_last_spks"]), -np.inf)
        self.all_spks_pyr: list[list[float]] = [[] for i in range(self.N_OUT)]
        self.rates_pattern_pyr: list[npt.NDArray] = []
        self.rates_buffer_pyr = np.zeros(self.N_OUT)

        self.validation = np.array([0], dtype=int)
        self.symmetrization = np.array([0], dtype=int)

        self.rng = rng

        self.w_up = np.zeros((self.N_OUT, self.N_IN))

        self.b_pyr = np.zeros_like(self.u_pyr)

        self.set_params(params)

    def set_params(self, params: dict[str, Any]) -> None:
        """Load synaptic and plasticity parameters from the parameter dict.

        Args:
            params: Network parameter dict.
        """
        self.n_last_spks = params["n_last_spks"]
        self.t_ref = params["t_ref"]
        self.tau_syn = params["tau_syn"]
        self.lam_nudge = params["lambda_nudge"]
        self.lr_up = params["lr"][self.LAYER_ID - 1]["up"]

    def set_bias(self, bias: npt.ArrayLike) -> None:
        """Set the bias for the output population.

        Args:
            bias: Bias vector for output neurons.
        """
        np.copyto(dst=self.b_pyr, src=bias)

    def set_weights_random(self):
        raise NotImplementedError

    def set_weights(self, w_up: npt.ArrayLike | None = None) -> None:
        """Copy ``w_up`` in-place if provided.

        Args:
            w_up: Bottom-up weights, shape ``(N_OUT, N_IN)``.
        """
        if w_up is not None:
            np.copyto(dst=self.w_up, src=w_up)

    def get_weights(self) -> dict[str, npt.NDArray]:
        """Return a copy of the bottom-up weight matrix.

        Returns:
            Dict with key ``"up"`` containing a copy of ``w_up``.
        """
        return {"up": self.w_up.copy()}

    def update_spks(self, t: float) -> None:
        """Sample output neuron spikes stochastically at time ``t``.

        Args:
            t: Current simulation time step.
        """
        r_pyr = logistic(self.u_pyr, self.t_ref)
        random_vals = self.rng.random(self.N_OUT)
        for i in np.nonzero(random_vals < r_pyr)[0]:
            if self.last_spks_pyr[i, -1] <= t - self.t_ref:
                if self.record_all_spks:
                    self.all_spks_pyr[i].append(t)
                self.last_spks_pyr[i, :-1] = self.last_spks_pyr[i, 1:]
                self.last_spks_pyr[i, -1] = t
                self.rates_buffer_pyr[i] += 1.0

    def calc_rates_pattern(self, t_pattern: float) -> None:
        """Append mean rates for the current pattern and reset the spike buffer.

        Args:
            t_pattern: Duration of the current pattern in time steps.
        """
        self.rates_pattern_pyr.append(self.rates_buffer_pyr / t_pattern)
        self.rates_buffer_pyr.fill(0.0)

    def dendritic_voltages(
        self, t: float, last_spks: npt.NDArray, weight: npt.NDArray
    ) -> npt.NDArray:
        """Compute weighted sum of rectangular PSPs from a spike history.

        Args:
            t: Current simulation time step.
            last_spks: Recent spike times, shape ``(n_pre, n_last_spks)``.
            weight: Synaptic weight matrix, shape ``(n_post, n_pre)``.

        Returns:
            Dendritic voltage contribution, shape ``(n_post,)``.
        """
        psps = np.sum(rect_psp(t - last_spks, self.tau_syn), axis=1)
        return np.dot(weight, psps)

    def update_mempot(
        self, t: float, last_spks_in: npt.NDArray, u_tgt: npt.NDArray | None
    ) -> None:
        """Update output membrane potential, optionally with target nudging.

        When ``u_tgt`` is not ``None``, the nudge compartment shifts the membrane
        potential towards the target via ``lam_nudge``.

        Args:
            t: Current simulation time step.
            last_spks_in: Recent spike times of the previous layer.
            u_tgt: Target membrane potential, or ``None`` during free (validation) phases.
        """
        self.v_bas[:] = self.dendritic_voltages(t, last_spks_in, self.w_up)
        if u_tgt is not None:
            self.v_nudge[:] = u_tgt - self.v_bas_av.val - self.b_pyr
            self.u_pyr[:] = self.b_pyr + self.v_bas + self.lam_nudge * self.v_nudge
            self.u_tgt[:] = u_tgt
        else:
            self.u_pyr[:] = self.b_pyr + self.v_bas
            self.u_tgt.fill(0.0)
        self.v_bas_av.move(self.v_bas)
        self.u_pyr_av.move(self.u_pyr)
        self.u_tgt_av.move(self.u_tgt)

    def plasticity_up(self, t: float, last_spks_in: npt.NDArray) -> None:
        """Apply RDD-based upward weight update.

        Args:
            t: Current simulation time step.
            last_spks_in: Recent spike times of the previous layer.
        """
        now_spks_in = np.where(last_spks_in[:, -1] == t - 1, 1, 0)
        r_bas_hat = logistic(self.v_bas_av.val + self.b_pyr, self.t_ref)
        r_pyr_hat = logistic(self.u_pyr_av.val, self.t_ref)
        delta_up = np.outer(r_pyr_hat - r_bas_hat, now_spks_in)
        self.w_up[:, :] += self.lr_up * delta_up

    def get_all_spks(self) -> dict[str, list[list[float]]]:
        """Return all recorded spike times for the output population.

        Returns:
            Dict with key ``"pyr"`` mapping to a list of per-neuron spike-time lists.
        """
        return {"pyr": self.all_spks_pyr}


class Network:
    """Layered network of cortical microcircuits (InputLayer → Layer... → OutputLayer)."""

    def __init__(self, params: dict[str, Any], poisson_seed: int) -> None:
        """Args:
        params: Network parameter dict (requires ``dims``, ``learning_lag``, ``t_ref``,
            ``lambda_api``, and all layer-level params).
        poisson_seed: Seed for the shared Poisson random number generator.
        """
        self.dims: list = params["dims"]
        self.n_dims: int = len(self.dims)
        self.len_learning_lag: int = params["learning_lag"]
        self.t_ref = params["t_ref"]
        self.lam_api = params["lambda_api"]

        self.update_up = True
        self.update_down = True

        self.p_rng = np.random.default_rng(poisson_seed)

        self.layers: list[Any] = [InputLayer(self.dims[0], params, self.p_rng)]
        for i in range(1, len(self.dims) - 1):
            self.layers.append(
                Layer(
                    n_pyr=self.dims[i],
                    n_in=self.dims[i - 1],
                    n_next=self.dims[i + 1],
                    params=params,
                    layer_id=i,
                    rng=self.p_rng,
                )
            )
        self.layers.append(
            OutputLayer(
                n_out=self.dims[-1],
                n_in=self.dims[-2],
                params=params,
                layer_id=i + 1,
                rng=self.p_rng,
            )
        )
        self.t = 0.0

    def set_weights(self, new_weights: WeightsList) -> None:
        """Set weights for all non-input layers from a ``WeightsList``.

        Args:
            new_weights: List of weight dicts, one per non-input layer.
        """
        assert (
            len(new_weights) == self.n_dims - 1
        ), f"new_weights should have {self.n_dims - 1} entries, but has {len(new_weights)}."  # noqa
        for i in range(1, self.n_dims):
            self.layers[i].set_weights(**new_weights[i - 1])

    def get_weights(self) -> WeightsList:
        """Return weight copies from all non-input layers.

        Returns:
            List of weight dicts, one per non-input layer.
        """
        return [layer.get_weights() for layer in self.layers[1:]]

    def set_bias(self, new_bias: list[npt.ArrayLike]) -> None:
        """Set biases for all non-input layers.

        Args:
            new_bias: List of bias arrays, one per non-input layer.
        """
        assert (
            len(new_bias) == self.n_dims - 1
        ), f"new_bias should have {self.n_dims - 1} entries, but has {len(new_bias)}."  # noqa
        for i in range(1, self.n_dims - 1):
            self.layers[i].set_bias(new_bias[i - 1], new_bias[i])
        self.layers[-1].set_bias(new_bias[-1])

    def set_sps(self) -> None:
        """Apply symmetric plasticity schedule (SPS) weights across all hidden layers."""
        for i in range(1, self.n_dims - 1):
            self.layers[i].set_sps(self.layers[i + 1].w_up)

    def set_copybackprop(self) -> None:
        """Set copy-back propagation weights across all hidden layers."""
        for i in range(1, self.n_dims - 1):
            self.layers[i].copybackprob(self.layers[i + 1].w_up)

    def set_record_all_spks(self, val: bool) -> None:
        """Enable or disable full spike recording in all layers.

        Args:
            val: ``True`` to record all spikes, ``False`` to disable.
        """
        for layer in self.layers:
            layer.record_all_spks = val

    def set_symmetrization(self) -> None:
        """Configure hidden layers for the symmetrization phase.

        Sets ``lam_api = 1`` and zeros out interneuron weights in all hidden layers.
        """
        for i in range(1, self.n_dims - 1):
            self.layers[i].lam_api = 1.0
            self.layers[i].set_weights(w_pi=0.0, w_ip=0.0)

    def unset_symmetrization(self) -> None:
        """Restore normal hidden-layer configuration after the symmetrization phase."""
        for i in range(1, self.n_dims - 1):
            self.layers[i].lam_api = self.lam_api
        self.set_sps()

    def distribute_weights(self, update_down: bool = True) -> None:
        """Initialize feedback weights; uses SPS or copy-back depending on ``update_down``.

        Args:
            update_down: If ``False``, use copy-back propagation instead of SPS.
        """
        if not update_down:
            self.set_copybackprop()
        self.set_sps()

    def init_tracker(
        self, rec_quants: list[list[str]], num_samples: int, compress_len: int
    ) -> None:
        """Allocate ``Tracker`` objects for the requested quantities and layers.

        Args:
            rec_quants: List of quantity name lists, one per layer (including input layer).
            num_samples: Number of compressed samples to store per tracker.
            compress_len: Number of time steps averaged into one stored sample.
        """
        assert len(rec_quants) == len(
            self.dims
        ), f"rec_quants should have {self.n_dims} entries, but has {len(rec_quants)}."  # noqa
        self.records = []
        # virual input layer:
        rcs = {}
        if "u_in" in rec_quants[0]:
            rcs["u_in"] = Tracker(num_samples, self.layers[0].u_in, compress_len)
        self.records.append(rcs)
        for i in range(1, self.n_dims - 1):
            rcs = {}
            if "u_pyr" in rec_quants[i]:
                rcs["u_pyr"] = Tracker(num_samples, self.layers[i].u_pyr, compress_len)
            if "v_bas" in rec_quants[i]:
                rcs["v_bas"] = Tracker(num_samples, self.layers[i].v_bas, compress_len)
            if "v_api" in rec_quants[i]:
                rcs["v_api"] = Tracker(num_samples, self.layers[i].v_api, compress_len)

            if "u_inn" in rec_quants[i]:
                rcs["u_inn"] = Tracker(num_samples, self.layers[i].u_inn, compress_len)

            if "w_up" in rec_quants[i]:
                rcs["w_up"] = Tracker(num_samples, self.layers[i].w_up, compress_len)
            if "w_pi" in rec_quants[i]:
                rcs["w_pi"] = Tracker(num_samples, self.layers[i].w_pi, compress_len)
            if "w_ip" in rec_quants[i]:
                rcs["w_ip"] = Tracker(num_samples, self.layers[i].w_ip, compress_len)
            if "w_down" in rec_quants[i]:
                rcs["w_down"] = Tracker(
                    num_samples, self.layers[i].w_down, compress_len
                )
            self.records.append(rcs)

        rcs = {}
        if "u_pyr" in rec_quants[-1]:
            rcs["u_pyr"] = Tracker(num_samples, self.layers[-1].u_pyr, compress_len)
        if "v_bas" in rec_quants[-1]:
            rcs["v_bas"] = Tracker(num_samples, self.layers[-1].v_bas, compress_len)
        if "v_nudge" in rec_quants[-1]:
            rcs["v_nudge"] = Tracker(num_samples, self.layers[-1].v_nudge, compress_len)

        if "u_tgt" in rec_quants[-1]:
            rcs["u_tgt"] = Tracker(num_samples, self.layers[-1].u_tgt, compress_len)

        if "w_up" in rec_quants[-1]:
            rcs["w_up"] = Tracker(num_samples, self.layers[-1].w_up, compress_len)

        if "validation" in rec_quants[-1]:
            rcs["validation"] = Tracker(
                num_samples, self.layers[-1].validation, compress_len
            )
        if "symmetrization" in rec_quants[-1]:
            rcs["symmetrization"] = Tracker(
                num_samples, self.layers[-1].symmetrization, compress_len
            )
        self.records.append(rcs)

    def update_spks(self) -> None:
        """Advance the spike state for all layers at the current time step."""
        for layer in self.layers:
            layer.update_spks(self.t)

    def update_mempot(self, u_in: npt.NDArray, u_tgt: npt.NDArray | None) -> None:
        """Update membrane potentials across all layers.

        Args:
            u_in: Input voltage for the input layer.
            u_tgt: Target voltage for the output layer, or ``None`` during validation.
        """
        self.layers[0].update_mempot(u_in)
        self.layers[1].update_mempot(
            self.t, self.layers[0].last_spks, self.layers[2].last_spks_pyr
        )
        for i in range(2, self.n_dims - 1):
            self.layers[i].update_mempot(
                self.t,
                self.layers[i - 1].last_spks_pyr,
                self.layers[i + 1].last_spks_pyr,
            )
        self.layers[-1].update_mempot(self.t, self.layers[-2].last_spks_pyr, u_tgt)

    def update_plasticity_up(self) -> None:
        """Apply upward plasticity updates to all eligible layers."""
        if self.update_up:
            self.layers[1].plasticity_up(self.t, self.layers[0].last_spks)
            for i in range(2, self.n_dims):
                self.layers[i].plasticity_up(self.t, self.layers[i - 1].last_spks_pyr)
        if not self.update_down:
            for i in range(1, self.n_dims - 1):
                self.layers[i].copybackprob(self.layers[i + 1].w_up)
        if self.set_sps_on:
            self.set_sps()

    def update_plasticity_down(self) -> None:
        """Apply STDP downward plasticity updates to all hidden layers."""
        if self.update_down:
            for i in range(1, self.n_dims - 1):
                self.layers[i].plasticity_down(self.t, self.layers[i + 1].last_spks_pyr)

    def update_step(
        self,
        u_in: npt.NDArray,
        u_tgt: npt.NDArray | None,
        plasticity_up_on: bool,
        plasticity_down_on: bool,
    ) -> None:
        """Perform one simulation time step: spikes → potentials → plasticity.

        Args:
            u_in: Input voltage for the input layer.
            u_tgt: Target voltage for the output layer, or ``None``.
            plasticity_up_on: Whether to apply upward plasticity updates.
            plasticity_down_on: Whether to apply downward plasticity updates.
        """
        self.update_spks()
        self.update_mempot(u_in, u_tgt)
        if plasticity_up_on:
            self.update_plasticity_up()
        if plasticity_down_on:
            self.update_plasticity_down()

    def record_quantities(self) -> None:
        """Call ``record()`` on all active ``Tracker`` objects."""
        for rcs in self.records:
            for quant in rcs.values():
                quant.record()

    def finalize_tracker(self) -> list[dict[str, npt.NDArray]]:
        """Flush all ``Tracker`` objects and return the recorded arrays.

        Returns:
            List of dicts (one per layer) mapping quantity names to compressed arrays.
        """
        tracker_res = []
        for rcs in self.records:
            res = {}
            for quant, tracker in rcs.items():
                tracker.finalize()
                res[quant] = tracker.data
            tracker_res.append(res)
        return tracker_res

    def finalize_mean_rates(self) -> list[dict[str, npt.NDArray]]:
        """Collect per-pattern mean spike rates from all layers.

        Returns:
            List of dicts (one per layer) mapping population names to rate arrays.
        """
        mean_spks = [{"input": np.array(self.layers[0].mean_rates_pattern)}]
        for layer in self.layers[1:-1]:
            mean_spks.append(
                {
                    "pyr": np.array(layer.rates_pattern_pyr),
                    "inn": np.array(layer.rates_pattern_inn),
                }
            )
        mean_spks.append({"pyr": np.array(self.layers[-1].rates_pattern_pyr)})
        return mean_spks

    def finalize_all_spks(self) -> list[dict[str, list[list[float]]]]:
        """Collect all spike trains from all layers.

        Returns:
            List of spike dicts (one per layer) as returned by each layer's ``get_all_spks()``.
        """
        return [layer.get_all_spks() for layer in self.layers]

    def calc_rates_pattern(self) -> None:
        """Finalize per-pattern rate estimates in all layers for the current pattern."""
        for layer in self.layers:
            layer.calc_rates_pattern(self.t_pattern)

    def run_pattern(
        self,
        u_inp: npt.NDArray,
        pattern_id: int,
        u_tgt: npt.NDArray | None = None,
        validation: bool = False,
        symmetrization: bool = False,
    ) -> None:
        """Run the network for one input pattern.

        Args:
            u_inp: Input voltage vector for this pattern.
            pattern_id: Index used to store the mean output in ``u_out_rec``.
            u_tgt: Target voltage for nudging; ``None`` disables nudging.
            validation: If ``True``, plasticity is suppressed regardless of ``u_tgt``.
            symmetrization: If ``True``, run the symmetrization phase for this pattern.
        """
        t_start = self.t
        t_stop = t_start + self.len_pattern
        # capture the mean output when the learning lag is over!
        u_out = np.zeros(self.dims[-1])
        if symmetrization:
            self.set_symmetrization()
        while self.t < t_stop:
            # for for len_learning_lag time-steps before palsticity is activated
            learning_lag = self.t - t_start >= self.len_learning_lag
            plasticity_up_on = (
                not validation
                and learning_lag
                and (u_tgt is not None)
                and not symmetrization
            )
            self.update_step(u_inp, u_tgt, plasticity_up_on, symmetrization)
            self.record_quantities()
            self.t += 1.0
            if learning_lag:
                u_out += self.layers[-1].u_pyr
        self.calc_rates_pattern()
        self.u_out_rec[pattern_id] = u_out / (self.len_pattern - self.len_learning_lag)
        if symmetrization:
            self.unset_symmetrization()

    def run(
        self,
        u_inp: npt.NDArray,
        t_pattern: int,
        recorded_quantities: list[list[str]],
        compress_len: int,
        len_epoch: int,
        validation_len: int,
        update_up: bool,
        update_down: bool,
        u_tgt: npt.NDArray | None = None,
        set_sps: bool = True,
        record_all_spks: bool = True,
        len_symm: int = 0,
    ) -> dict[str, Any]:
        """Run the network for the full simulation and return results.

        Args:
            u_inp: Input voltages, shape ``(n_patterns, n_in)``.
            t_pattern: Number of refractory periods per pattern.
            recorded_quantities: Quantity names to record, one list per layer.
            compress_len: Time steps averaged into one recorded sample.
            len_epoch: Number of training patterns per epoch.
            validation_len: Number of validation patterns per epoch.
            update_up: Whether to apply upward plasticity.
            update_down: Whether to apply downward plasticity.
            u_tgt: Target voltages, shape ``(n_patterns, n_out)``, or ``None``.
            set_sps: Whether to update SPS weights after each upward update.
            record_all_spks: Whether to record the full spike trains.
            len_symm: Number of symmetrization patterns at the start of each epoch.

        Returns:
            Dict with keys ``"recordings"``, ``"mean_rates"``, ``"all_spks"``,
            ``"u_out"``, ``"u_in"``, ``"u_tgt"``, and ``"validation"``.
        """
        self.t_pattern: int = t_pattern
        self.len_pattern: int = t_pattern * self.t_ref
        assert (
            self.len_learning_lag < self.len_pattern
        ), "The learning lag must be smaller than the presentation time of a pattern."
        assert (
            len_symm < len_epoch
        ), "The symmetrization length must be smaller than the epoch len."

        self.update_up = update_up
        self.update_down = update_down
        self.set_sps_on = set_sps
        if isinstance(u_tgt, np.ndarray):
            assert u_inp.shape[0] == u_tgt.shape[0]
        else:
            u_tgt = repeat(None)

        num_rec_samples = int(np.ceil(len(u_inp) * self.len_pattern / compress_len))
        self.init_tracker(recorded_quantities, num_rec_samples, compress_len)
        self.set_record_all_spks(record_all_spks)

        total_epoch_len = len_epoch + validation_len

        self.u_out_rec = np.empty((u_inp.shape[0], self.dims[-1]))

        validation_loss = []
        validation_times = []
        realtime_start = time.time()
        for pat_id, (u_pattern, u_target) in enumerate(zip(u_inp, u_tgt)):
            i_epoch = pat_id // total_epoch_len
            if pat_id % total_epoch_len == 0:
                print(f"### EPOCH NO. {i_epoch + 1} ###")
            print(f"Pattern no. {pat_id + 1} / {len(u_inp)}")
            validation_counter = pat_id - i_epoch * total_epoch_len - len_epoch
            symm_counter = validation_counter + len_symm

            if symm_counter == 0:
                print("### SYMMETRIZATION ###")
            self.layers[-1].symmetrization[0] = (
                symm_counter >= 0 and validation_counter < 0
            )

            if validation_counter == 0:
                print("### VALIDATION ###")
                intermediate_validation_loss = 0
                validation_times.append(pat_id)
            self.layers[-1].validation[0] = validation_counter >= 0

            self.run_pattern(
                u_pattern,
                pat_id,
                u_target,
                self.layers[-1].validation[0],
                symmetrization=self.layers[-1].symmetrization[0],
            )
            if 0 <= validation_counter < validation_len:
                if u_target is not None:
                    intermediate_validation_loss += loss_func(
                        self.u_out_rec[pat_id], u_target
                    )
            if validation_counter == validation_len - 1:
                validation_loss.append(intermediate_validation_loss / validation_len)

            realtime_now = time.time()
            av_time_per_pattern = (realtime_now - realtime_start) / (pat_id + 1)
            av_time_per_dt = av_time_per_pattern / self.len_pattern
            print(
                f"Average time per pattern {av_time_per_pattern:.1f} s, average time per dt {(av_time_per_dt * 1000):.2f} ms"  # noqa
            )
            print(f"ETA: {(av_time_per_pattern * (len(u_inp) - pat_id - 1)):.1f}")

        res = {
            "recordings": self.finalize_tracker(),
            "mean_rates": self.finalize_mean_rates(),
            "all_spks": self.finalize_all_spks(),
            "u_out": self.u_out_rec,
            "u_in": u_inp,
            "u_tgt": u_tgt,
            "validation": {"loss": validation_loss, "times": validation_times},
        }

        return res
