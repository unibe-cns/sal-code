"""Neural sampling implementation (Buesing et al. 2011)."""

import time
from datetime import datetime
from typing import Any, Callable, TypeAlias

import numba
import numpy as np
import numpy.typing as npt

from .utils import (
    bm_to_probs,
    distr_from_states,
    get_states_from_spikes,
    list_of_states,
    ordered_spikes_to_list,
)

# declare my own types here:
StdpFunc: TypeAlias = Callable[[npt.ArrayLike], npt.ArrayLike]
SimParams: TypeAlias = dict[str, Any]


@numba.njit(cache=False)
def logistic(x: float, t_ref: float) -> float:
    """Logistic activation function shifted by log(t_ref)."""
    return 1.0 / (1.0 + np.exp(-(x - np.log(t_ref))))


@numba.vectorize(cache=False)
def heaviside(x: float) -> float:
    """Heaviside step function: 1 if x > 0, else 0."""
    if x > 0.0:
        return 1.0
    else:
        return 0.0


@numba.njit(cache=False)
def rect_kernel(x: float, tau_syn: float) -> float:
    """Rectangular PSP kernel: 1 within (0, tau_syn], else 0."""
    return heaviside(x) * heaviside(-x + tau_syn)


@numba.njit(cache=False)
def alpha_kernel(x: float, tau_ref: float, tau_syn: float) -> float:
    """Alpha-function PSP kernel."""
    return heaviside(tau_ref / tau_syn**2 * x * np.exp(-x / tau_syn))


@numba.njit(cache=False)
def calc_inst_rate(
    t: int,
    last_spikes: npt.NDArray,
    bias: npt.NDArray,
    weight_mat: npt.NDArray,
    t_ref: float,
    tau_syn: float,
    psp_kernel: Callable,
) -> npt.NDArray:
    """Compute instantaneous firing rates for all neurons at time t.

    Args:
        t: Current time step.
        last_spikes: (N, K) array of the K most recent spike times per neuron.
        bias: Neuron bias vector.
        weight_mat: Synaptic weight matrix.
        t_ref: Refractory period.
        tau_syn: Synaptic time constant.
        psp_kernel: PSP kernel function.

    Returns:
        Array of instantaneous firing rates, one per neuron.
    """
    psps = np.sum(psp_kernel(t - last_spikes, tau_syn), axis=1)
    mem_pot = bias + np.dot(weight_mat, psps)
    inst_rate = logistic(mem_pot, t_ref)
    # to match with buesing: substract log(t_ref) from mem_pot
    # (mind the unit of t_ref!)
    return inst_rate


@numba.njit(cache=False)
def sim_poisson_neurons(
    t_max: int,
    psp_kernel: Callable,
    bias: npt.NDArray,
    weights: npt.NDArray,
    t_ref: float,
    tau_syn: float,
    num_last_spikes: int = 10,
) -> npt.NDArray:
    """Simulate Poisson spiking neurons with PSP-based interactions.

    Args:
        t_max: Simulation duration in time steps.
        psp_kernel: PSP kernel function.
        bias: Neuron bias vector.
        weights: Synaptic weight matrix.
        t_ref: Refractory period.
        tau_syn: Synaptic time constant.
        num_last_spikes: Number of past spikes tracked per neuron.

    Returns:
        Ordered spike array of (time, neuron_id) tuples.
    """
    num_neurons = len(bias)

    ordered_spikes = []
    # how many past spikes do we take into account for the calculation of the
    # PSPs
    last_spikes = np.full((num_neurons, num_last_spikes), -100_000_000)

    for t in range(t_max):
        inst_rate = calc_inst_rate(
            t, last_spikes, bias, weights, t_ref, tau_syn, psp_kernel
        )
        # probability to spike in [t, t+dt]
        random_vals = np.random.random_sample(num_neurons)
        # check if the prob is smaller than the random value
        for i in np.nonzero(random_vals < inst_rate)[0]:
            # refractory machanism:
            if last_spikes[i, -1] < t - t_ref:
                # use nest/ssn convention
                ordered_spikes.append((t, i + 1.0))
                # push back the last_spikes-stack
                last_spikes[i, :-1] = last_spikes[i, 1:]
                last_spikes[i, -1] = t

    return np.array(ordered_spikes)


class NeuralSampler:
    """Base neural sampler with wake and sleep phase simulation.

    Args:
        init_weight: Initial weight matrix (N, N).
        init_bias: Initial bias vector (N,).
        num_visible: Number of visible neurons.
        sim_params: Dict with keys psp_kernel, t_ref, tau_syn, num_last_spikes.
        rng_seed: Random seed.
    """

    def __init__(
        self,
        init_weight: npt.NDArray,
        init_bias: npt.NDArray,
        num_visible: int,
        sim_params: SimParams,
        rng_seed: int = 424242,
    ) -> None:
        np.random.seed(rng_seed)

        assert init_weight.shape[0] == init_weight.shape[1]
        assert init_weight.shape[0] == len(init_bias)

        self.num_nrns = len(init_bias)
        self.num_vis = num_visible
        self.num_hidden = self.num_nrns - self.num_vis
        assert self.num_hidden >= 0
        self.weight = init_weight
        self.bias = init_bias

        self.psp_kernel = sim_params["psp_kernel"]
        self.t_ref = sim_params["t_ref"]
        self.tau_syn = sim_params["tau_syn"]
        self.num_last_spikes = sim_params["num_last_spikes"]
        self.rng_seed = rng_seed

    def wake_phase(self, sim_dur: int, target: npt.NDArray) -> npt.NDArray:
        """Run a clamped simulation with target clamped to visible neurons."""
        bias = np.copy(self.bias)
        bias[: self.num_vis] = (target * 2.0 - 1.0) * 10.0
        spikes = sim_poisson_neurons(
            sim_dur, self.psp_kernel, bias, self.weight, self.t_ref, self.tau_syn
        )

        return spikes

    def sleep_phase(self, sim_dur: int) -> npt.NDArray:
        """Run a free simulation for `sim_dur` steps."""
        spikes = sim_poisson_neurons(
            sim_dur, self.psp_kernel, self.bias, self.weight, self.t_ref, self.tau_syn
        )
        return spikes

    def spikes_to_states(self, spikes: npt.NDArray, sim_dur: int) -> npt.NDArray:
        """Convert a spike array to a binary state matrix."""
        t_refs = np.full(self.num_nrns, self.t_ref)
        return get_states_from_spikes(
            self.num_nrns, spikes, t_refs, self.t_ref / 2.0, sim_dur
        )

    def restrict(self, arr: npt.NDArray) -> npt.NDArray:
        """Zero out visible-visible and hidden-hidden entries of arr."""
        arr[: self.num_vis, : self.num_vis] = 0.0
        arr[self.num_vis :, self.num_vis :] = 0.0
        return arr

    def restrict_weights(self) -> None:
        """Zero out visible-visible and hidden-hidden weights in-place."""
        self.weight[: self.num_vis, : self.num_vis] = 0.0
        self.weight[self.num_vis :, self.num_vis :] = 0.0

    def clip_weights(self, max_w: float) -> None:
        """Clip all weights to [-max_w, max_w] in-place."""
        self.weight[self.weight > max_w] = max_w
        self.weight[self.weight < -max_w] = -max_w

    def clip_bias(self, max_b: float) -> None:
        """Clip all biases to [-max_b, max_b] in-place."""
        self.bias[self.bias > max_b] = max_b
        self.bias[self.bias < -max_b] = -max_b


class NeuralSamplerFullyConnected(NeuralSampler):
    """Fully connected neural sampler trained with STDP-based gradient descent.

    Args:
        init_weight: Initial weight matrix.
        init_bias: Initial bias vector.
        target_weight: Target BM weight matrix (used to compute target distribution).
        target_bias: Target BM bias vector.
        sim_params: Simulation parameters dict.
        dur_sleep: Sleep phase duration.
        optimizer_bias: Bias update optimizer.
        optimizer_weight: Weight update optimizer.
        optimizer_symm: Symmetrization optimizer (optional).
        max_w: Weight clip bound.
        max_b: Bias clip bound.
        rng_seed: Random seed.
        weight_decay: Per-step weight decay fraction.
    """

    def __init__(
        self,
        init_weight: npt.NDArray,
        init_bias: npt.NDArray,
        target_weight: npt.NDArray,
        target_bias: npt.NDArray,
        sim_params: SimParams,
        dur_sleep: int,
        optimizer_bias: Callable,
        optimizer_weight: Callable,
        optimizer_symm: Callable | None = None,
        max_w: float = 2.0,
        max_b: float = 2.0,
        rng_seed: int = 424242,
        weight_decay: float | npt.NDArray = 0.0,
    ):
        """Initialize sampler and compute target distribution analytically."""
        super().__init__(init_weight, init_bias, 0, sim_params, rng_seed=rng_seed)

        self.dur = dur_sleep
        self.optimizer_bias = optimizer_bias
        self.optimizer_weight = optimizer_weight
        self.optimizer_symm = optimizer_symm
        self.los = list_of_states(self.num_nrns)
        self.max_w = max_w
        self.max_b = max_b
        self.validation = False

        self.weight_decay = 1.0 - weight_decay

        self.target_distr, self.los, self.coact = bm_to_probs(
            target_weight, target_bias
        )

        self.marginals = np.diagonal(self.coact).copy()
        np.fill_diagonal(self.coact, 0.0)
        print("theoretical distribution", self.target_distr, flush=True)
        print("theoretical coactivation \n", self.coact, flush=True)
        print("theoretical marginals", self.marginals, flush=True)

        print(self.target_distr.shape)

    def spike_rates(self, spikes: npt.NDArray, dur: float | int) -> npt.NDArray:
        """Compute mean spike rates for all neurons.

        Args:
            spikes: Ordered spike array.
            dur: Simulation duration.

        Returns:
            Rate array, one per neuron.
        """
        rates = []
        list_of_spikes = ordered_spikes_to_list(
            spikes, list(range(1, self.num_nrns + 1))
        )
        for spks in list_of_spikes:
            rates.append(len(spks) / dur * self.t_ref)
        return np.array(rates)

    def sleep_phase(
        self, stdp_rule: StdpFunc, sal_rule: StdpFunc | None = None
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray | None]:
        """Run a sleep phase and return STDP, rates, sampled distribution, and SAL.

        Args:
            stdp_rule: STDP rule callable.
            sal_rule: SAL rule callable (optional).

        Returns:
            Tuple of (stdp, rates, sampled_distr, stdp_sal).
        """

        spikes = super().sleep_phase(self.dur)
        states = self.spikes_to_states(spikes, self.dur)
        sampled_distr = distr_from_states(states, self.los)

        rates = self.spike_rates(spikes, self.dur)

        stdp = stdp_rule(spikes) / self.dur * self.t_ref

        if sal_rule is not None:
            stdp_sal = sal_rule(spikes) / self.dur * self.t_ref
        else:
            stdp_sal = None

        return stdp, rates, sampled_distr, stdp_sal

    def training_iteration(
        self,
        stdp_rule: StdpFunc,
        sal_rule: StdpFunc | None = None,
    ) -> dict:
        """Run one training step: sleep phase → gradient → weight/bias update.

        Args:
            stdp_rule: STDP rule callable.
            sal_rule: SAL rule callable (optional).

        Returns:
            Dict with sampled_distr, weights, biases, sleep_stdp, sal_stdp, target_distr.
        """
        sleep_stdp, sleep_rates, sampled_distr, stdp_sal = self.sleep_phase(
            stdp_rule=stdp_rule,
            sal_rule=sal_rule,
        )

        corrected_coact = (
            self.coact * stdp_rule.noised_correlation_factors() / self.t_ref
        )
        grad_weight = corrected_coact - sleep_stdp
        grad_bias = self.marginals - sleep_rates

        delta_weight = self.optimizer_weight.update(grad_weight)
        delta_bias = self.optimizer_bias.update(grad_bias)

        # update params:
        self.bias = self.bias + delta_bias
        self.weight = self.weight + delta_weight

        # weight decay
        self.weight = self.weight * self.weight_decay

        # optional: symmetrization with sal
        if sal_rule is not None:
            delta_sal = self.optimizer_symm(stdp_sal)
            self.weight = self.weight + delta_sal

        # impose RBM restrictions:
        self.clip_weights(self.max_w)
        self.clip_bias(self.max_b)
        np.fill_diagonal(self.weight, 0.0)

        res = {
            "sampled_distr": sampled_distr,
            "weights": np.copy(self.weight),
            "biases": np.copy(self.bias),
            "sleep_stdp": sleep_stdp,
            "sal_stdp": stdp_sal,
            "target_distr": self.target_distr,
        }

        return res

    def train(
        self,
        num_iter: int,
        stdp_rule: StdpFunc,
        stdp_rule_symm: StdpFunc | None = None,
        callback: Callable | None = None,
        validation_step: int = 1,
        validation_factor: int = 10,
    ) -> None:
        """Train for `num_iter` iterations, calling `callback` after each step.

        Args:
            num_iter: Number of training iterations.
            stdp_rule: STDP rule callable.
            stdp_rule_symm: SAL rule callable (optional).
            callback: Called with (result_dict, step); return True to stop early.
            validation_step: Run a longer validation every this many steps.
            validation_factor: Multiply sleep duration during validation.
        """
        for step in range(num_iter):
            # every validation_step-th iteration change the sleep duration,
            # but not if validation_step == 1
            if step % validation_step == 0 and (validation_step - 1):
                tick = time.time()
                self.dur *= validation_factor
                print(
                    f"TRAINING ITERATION NO. {step} -- {datetime.now().ctime()}",
                    flush=True,
                )
                print("validation phase!")
                self.validation = True
            res = self.training_iteration(
                stdp_rule,
                sal_rule=stdp_rule_symm,
            )

            quit = False
            if callback is not None:
                quit = callback(res, step)

            if step % validation_step == 0 and (validation_step - 1):
                tock = time.time()
                self.dur /= validation_factor
                self.validation = False
                print(f"Time for iteration: {tock - tick}", flush=True)

            if quit:
                print(
                    "Recieved signal to stop training from callback function!",
                    flush=True,
                )
                break


class GradDescent(object):
    """Gradient descent optimizer: update = lr * grad."""

    def __init__(self, lr: float) -> None:
        """Set learning rate."""
        self.lr = lr

    def update(self, grad: npt.NDArray) -> npt.NDArray:
        """Return lr * grad."""
        return self.lr * grad

    def __call__(self, grad: npt.NDArray) -> npt.NDArray:
        """Alias for `update`."""
        return self.update(grad)
