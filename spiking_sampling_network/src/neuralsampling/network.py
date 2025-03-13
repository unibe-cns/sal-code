"""
Implement neural sampling (Buesing et al. 2011)
"""

import multiprocessing as mp
import os
import time
from datetime import datetime
from itertools import repeat
from typing import Any, Callable, Optional, Tuple, TypeAlias

import numba
import numpy as np
import numpy.typing as npt

from .utils import (
    bm_to_probs,
    calc_coact_from_states,
    calc_dkl,
    distr_from_states,
    get_states_from_spikes,
    list_of_states,
    ordered_spikes_to_list,
    prettyprint_distribution,
)

# declare my own types here:
StdpFunc: TypeAlias = Callable[[npt.ArrayLike], npt.ArrayLike]
SimParams: TypeAlias = dict[str, Any]


@numba.njit(cache=False)
def logistic(x, t_ref):
    return 1.0 / (1.0 + np.exp(-(x - np.log(t_ref))))


@numba.vectorize(cache=False)
def heaviside(x):
    if x > 0.0:
        return 1.0
    else:
        return 0.0


@numba.njit(cache=False)
def rect_kernel(x, tau_syn):
    return heaviside(x) * heaviside(-x + tau_syn)


@numba.njit(cache=False)
def alpha_kernel(x, tau_ref, tau_syn):
    return heaviside(tau_ref / tau_syn**2 * x * np.exp(-x / tau_syn))


@numba.njit(cache=False)
def calc_inst_rate(t, last_spikes, bias, weight_mat, t_ref, tau_syn, psp_kernel):
    psps = np.sum(psp_kernel(t - last_spikes, tau_syn), axis=1)
    mem_pot = bias + np.dot(weight_mat, psps)
    # TODO: Give the activation function parameters and noise them!
    inst_rate = logistic(mem_pot, t_ref)
    # to match with buesing: substract log(t_ref) from mem_pot
    # (mind the unit of t_ref!)
    return inst_rate


@numba.njit(cache=False)
def sim_poisson_neurons(
    t_max, psp_kernel, bias, weights, t_ref, tau_syn, num_last_spikes=10
):
    num_neurons = len(bias)

    ordered_spikes = []
    # how many past spikes do we take into account for the calculation of the
    # PSPs
    last_spikes = np.full((num_neurons, num_last_spikes), -np.inf)

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
    def __init__(
        self, init_weight, init_bias, num_visible, sim_params, rng_seed=424242
    ):
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

    def wake_phase(self, sim_dur, target):
        bias = np.copy(self.bias)
        bias[: self.num_vis] = (target * 2.0 - 1.0) * 10.0
        spikes = sim_poisson_neurons(
            sim_dur, self.psp_kernel, bias, self.weight, self.t_ref, self.tau_syn
        )

        return spikes

    def sleep_phase(self, sim_dur):
        spikes = sim_poisson_neurons(
            sim_dur, self.psp_kernel, self.bias, self.weight, self.t_ref, self.tau_syn
        )
        return spikes

    def spikes_to_states(self, spikes, sim_dur):
        t_refs = np.full(self.num_nrns, self.t_ref)
        return get_states_from_spikes(
            self.num_nrns, spikes, t_refs, self.t_ref / 2.0, sim_dur
        )

    def restrict(self, arr):
        arr[: self.num_vis, : self.num_vis] = 0.0
        arr[self.num_vis :, self.num_vis :] = 0.0
        return arr

    def restrict_weights(self):
        self.weight[: self.num_vis, : self.num_vis] = 0.0
        self.weight[self.num_vis :, self.num_vis :] = 0.0

    def clip_weights(self, max_w):
        self.weight[self.weight > max_w] = max_w
        self.weight[self.weight < -max_w] = -max_w

    def clip_bias(self, max_b):
        self.bias[self.bias > max_b] = max_b
        self.bias[self.bias < -max_b] = -max_b


class NeuralSamplerTrainDistr(NeuralSampler):
    def __init__(
        self,
        init_weight,
        init_bias,
        target_weight,
        target_bias,
        sim_params,
        sim_dur,
        optimizer_weight,
        optimizer_bias,
        rng_seed=424242,
    ):
        self.sim_dur = sim_dur
        self.target_weight = target_weight
        self.target_bias = target_bias
        self.target_distr, self.los, self.target_coact = bm_to_probs(
            self.target_weight, self.target_bias
        )
        # los = list of states
        self.target_marginals = np.diagonal(self.target_coact).copy()
        np.fill_diagonal(self.target_coact, 0.0)
        print("theoretical distribution", self.target_distr, flush=True)
        print("theoretical coactivation \n", self.target_coact, flush=True)
        print("theoretical marginals", self.target_marginals, flush=True)

        self.optimizer_weight = optimizer_weight
        self.optimizer_bias = optimizer_bias

        super().__init__(init_weight, init_bias, 0, sim_params, rng_seed=rng_seed)

    def training_iteration(self):
        spikes = self.sleep_phase(self.sim_dur)
        states = self.spikes_to_states(spikes, self.sim_dur)

        sampled_distr = distr_from_states(states, self.los)

        prettyprint_distribution(self.los, sampled_distr, self.target_distr)
        dkl = calc_dkl(self.target_distr, sampled_distr)
        print(f"DKL to target: {dkl}", flush=True)

        sampled_coact, sampled_marg = calc_coact_from_states(states)
        print("sampled coact \n", sampled_coact, flush=True)
        print("sampled marginals \n", sampled_marg, flush=True)

        grad_bias = self.target_marginals - sampled_marg
        grad_weight = self.target_coact - sampled_coact

        delta_bias = self.optimizer_bias(grad_bias)
        delta_weight = self.optimizer_weight(grad_weight)

        # update params:
        self.bias = self.bias + delta_bias
        self.weight = self.weight + delta_weight

        res = {
            "sampled_distr": sampled_distr,
            "dkl": dkl,
        }

        return res

    def train(self, num_iterations, callback=None):
        """
        callback-function:
            Signature: func(dict: res, int: step)
                res = intermediate result from one training iteration
                step = number of training iteration
            If callback function returns anything that evaluates as True,
            stop the training (i.e. if certain DKL is reached).
        """

        for step in range(num_iterations):
            print(f"TRAINING ITERATION NO. {step}")

            tick = time.time()
            res = self.training_iteration()
            quit = False
            if callback is not None:
                quit = callback(res, step)

            if quit:
                print(
                    "Recieved signal to stop training from callback function!",
                    flush=True,
                )
                break

            tock = time.time()
            print(f"Time for iteration: {tock - tick}", flush=True)


class NeuralSamplerStateTraining(NeuralSampler):
    """docstring for NeuralSamplerStateTraining."""

    def __init__(
        self,
        init_weight,
        init_bias,
        num_visible,
        sim_params,
        sim_dur,
        batchsize,
        optimizer_bias,
        optimizer_weight,
        rng_seed=424242,
    ):
        super().__init__(
            init_weight, init_bias, num_visible, sim_params, rng_seed=rng_seed
        )

        self.sim_dur = sim_dur
        self.batchsize = batchsize
        self.optimizer_bias = optimizer_bias
        self.optimizer_weight = optimizer_weight
        self.los = list_of_states(self.num_vis)

        # impose resctrictions on the weight matrix:
        self.restrict_weights()

    def single_wake_phase(self, pattern):
        spikes = self.wake_phase(self.sim_dur, pattern)
        states = self.spikes_to_states(spikes, self.sim_dur)
        sampled_coact, sampled_marg = calc_coact_from_states(states)
        return sampled_coact, sampled_marg

    def batch_wake_phase(self, pattern_batch):
        assert pattern_batch.shape == (self.batchsize, self.num_vis)

        batch_coact = np.zeros_like(self.weight)
        batch_marg = np.zeros_like(self.bias)

        for i in range(self.batchsize):
            sampled_coact, sampled_marg = self.single_wake_phase(pattern_batch[i])
            batch_coact += sampled_coact
            batch_marg += sampled_marg

        batch_marg /= self.batchsize
        batch_coact /= self.batchsize

        return batch_coact, batch_marg

    def batch_wake_phase_parallel(self, pattern_batch, n_procs=os.cpu_count()):
        if n_procs > self.batchsize:
            n_procs == self.batchsize

        with mp.Pool(processes=n_procs) as pool:
            res = pool.map(self.single_wake_phase, pattern_batch)

        batch_coact = np.zeros_like(self.weight)
        batch_marg = np.zeros_like(self.bias)

        for coact, marg in res:
            batch_coact += coact
            batch_marg += marg

        batch_marg /= self.batchsize
        batch_coact /= self.batchsize

        return batch_coact, batch_marg

    def single_sleep_phase(self):
        tick = time.time()
        spikes = self.sleep_phase(self.sim_dur)
        print("sampler time", time.time() - tick)
        tick = time.time()
        states = self.spikes_to_states(spikes, self.sim_dur)
        print("spikes to states time", time.time() - tick)
        tick = time.time()
        sampled_coact, sampled_marg = calc_coact_from_states(states)
        print("calc coact time", time.time() - tick)
        tick = time.time()
        sampled_distr = distr_from_states(states[:, : self.num_vis], self.los)
        print("distr from states time", time.time() - tick)

        return sampled_coact, sampled_marg, sampled_distr

    def training_iteration(self, pattern_batch, n_procs=1):
        if n_procs == 1:
            print("one process")
            wake_coact, wake_marg = self.batch_wake_phase(pattern_batch)
        else:
            wake_coact, wake_marg = self.batch_wake_phase_parallel(
                pattern_batch, n_procs=n_procs
            )

        sleep_coact, sleep_marg, sampled_distr = self.single_sleep_phase()

        grad_weight = wake_coact - sleep_coact
        grad_bias = wake_marg - sleep_marg

        delta_weight = self.optimizer_weight(grad_weight)
        delta_bias = self.optimizer_bias(grad_bias)

        # update params:
        self.bias = self.bias + delta_bias
        self.weight = self.weight + delta_weight

        # impose RBM restrictions:
        self.restrict_weights()

        res = {"sampled_distr": sampled_distr}

        return res

    def train(self, num_batches, pattern, n_procs=os.cpu_count(), callback=None):
        assert num_batches * self.batchsize <= len(pattern)

        print(f"Number of processes: {n_procs}", flush=True)
        for step in range(num_batches):
            print(f"TRAINING ITERATION NO. {step}")

            tick = time.time()
            # pick a batch:
            batch = pattern[step * self.batchsize : (step + 1) * self.batchsize]
            res = self.training_iteration(batch, n_procs=n_procs)

            quit = False
            if callback is not None:
                quit = callback(res, step)

            if quit:
                print(
                    "Recieved signal to stop training from callback function!",
                    flush=True,
                )
                break
            tock = time.time()
            print(f"Time for iteration: {tock - tick}", flush=True)


class NeuralSamplerSpikeTraining(NeuralSampler):
    """docstring for NeuralSamplerSpikeTraining."""

    def __init__(
        self,
        init_weight,
        init_bias,
        num_visible,
        sim_params,
        dur_wake,
        dur_sleep,
        batchsize,
        optimizer_bias,
        optimizer_weight,
        optimizer_symm=None,
        max_w=1.0,
        max_b=1.0,
        rng_seed=424242,
    ):
        super().__init__(
            init_weight, init_bias, num_visible, sim_params, rng_seed=rng_seed
        )
        self.dur_wake = dur_wake
        self.dur_sleep = dur_sleep
        self.batchsize = batchsize
        self.optimizer_bias = optimizer_bias
        self.optimizer_weight = optimizer_weight
        self.optimizer_symm = optimizer_symm
        self.los = list_of_states(self.num_vis)
        self.max_b = max_b
        self.max_w = max_w

        # impose resctrictions on the weight matrix:
        self.restrict_weights()
        self.clip_bias(max_b)
        self.clip_weights(max_w)

    def spike_rates(self, spikes, dur):
        rates = []
        list_of_spikes = ordered_spikes_to_list(
            spikes, list(range(1, self.num_nrns + 1))
        )
        for spks in list_of_spikes:
            rates.append(len(spks) / dur * self.t_ref)
        return np.array(rates)

    def single_wake_phase(self, pattern, stdp_rule):
        spikes = self.wake_phase(self.dur_wake, pattern)
        stdp = stdp_rule(spikes) / self.dur_wake * self.t_ref
        rates = self.spike_rates(spikes, self.dur_wake)
        return stdp, rates

    def batch_wake_phase(self, pattern_batch, stdp_rule):
        assert pattern_batch.shape == (self.batchsize, self.num_vis)

        batch_stdp = np.zeros_like(self.weight)
        batch_rates = np.zeros_like(self.bias)

        for i in range(self.batchsize):
            sampled_stdp, sampled_rates = self.single_wake_phase(
                pattern_batch[i], stdp_rule
            )
            batch_stdp += sampled_stdp
            batch_rates += sampled_rates

        batch_stdp /= self.batchsize
        batch_rates /= self.batchsize

        return batch_stdp, batch_rates

    def batch_wake_phase_parallel(
        self, pattern_batch, stdp_rule, n_procs=os.cpu_count()
    ):
        if n_procs > self.batchsize:
            n_procs == self.batchsize

        params = zip(pattern_batch, repeat(stdp_rule))
        with mp.Pool(processes=n_procs) as pool:
            res = pool.starmap(self.single_wake_phase, params)

        batch_stdp = np.zeros_like(self.weight)
        batch_rates = np.zeros_like(self.bias)

        for stdp, rates in res:
            batch_stdp += stdp
            batch_rates += rates

        batch_stdp /= self.batchsize
        batch_rates /= self.batchsize

        return batch_stdp, batch_rates

    def batch_sleep_phase(self) -> None:
        """TODO."""
        raise NotImplementedError()

    def batch_sleep_phase_parallel(self) -> None:
        """TODO.

        Run several sleep phase workers in parallel and combine them to get more
        samples
        """
        raise NotImplementedError()

    def single_sleep_phase(
        self, stdp_rule: StdpFunc, sal_rule: Optional[StdpFunc] = None
    ) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray, Optional[npt.NDArray]]:
        spikes = self.sleep_phase(self.dur_sleep)
        states = self.spikes_to_states(spikes, self.dur_sleep)
        sampled_distr = distr_from_states(states[:, : self.num_vis], self.los)

        rates = self.spike_rates(spikes, self.dur_sleep)

        stdp = stdp_rule(spikes) / self.dur_sleep * self.t_ref

        if sal_rule is not None:
            stdp_sal = sal_rule(spikes) / self.dur_sleep * self.t_ref
        else:
            stdp_sal = None
        return stdp, rates, sampled_distr, stdp_sal

    def training_iteration(
        self,
        pattern_batch: npt.NDArray,
        stdp_rule: StdpFunc,
        stdp_rule_sleep: Optional[StdpFunc] = None,
        sal_rule: Optional[StdpFunc] = None,
        n_procs=1,
    ) -> dict:
        if n_procs == 1:
            print("one process")
            wake_stdp, wake_rates = self.batch_wake_phase(pattern_batch, stdp_rule)
        else:
            wake_stdp, wake_rates = self.batch_wake_phase_parallel(
                pattern_batch, stdp_rule, n_procs=n_procs
            )

        sleep_stdp, sleep_rates, sampled_distr, stdp_sal = self.single_sleep_phase(
            stdp_rule=stdp_rule_sleep if stdp_rule_sleep is not None else stdp_rule,
            sal_rule=sal_rule,
        )

        grad_weight = wake_stdp - sleep_stdp
        grad_bias = wake_rates - sleep_rates

        delta_weight = self.optimizer_weight.update(grad_weight)
        delta_bias = self.optimizer_bias.update(grad_bias)

        # update params:
        self.bias = self.bias + delta_bias
        self.weight = self.weight + delta_weight

        # optional: symmetrization with sal
        if sal_rule is not None:
            delta_sal = self.optimizer_symm(stdp_sal)
            self.weight = self.weight + delta_sal

        # impose RBM restrictions:
        self.restrict_weights()
        self.clip_weights(self.max_w)
        self.clip_bias(self.max_b)

        res = {
            "sampled_distr": sampled_distr,
            "weights": np.copy(self.weight),
            "biases": np.copy(self.bias),
            "wake_stdp": wake_stdp,
            "sleep_stdp": sleep_stdp,
            "sal_stdp": stdp_sal,
        }

        return res

    def symmetrization_iteration(self, stdp_rule):
        stdp, rates, sampled_distr, _ = self.single_sleep_phase(stdp_rule)

        delta_weight = self.optimizer_symm(stdp)

        self.weight = self.weight + delta_weight
        self.restrict_weights()

        res = {
            "sampled_distr": sampled_distr,
            "weights": np.copy(self.weight),
            "biases": np.copy(self.bias),
        }

        return res

    def train(
        self,
        num_batches,
        pattern,
        stdp_rule_train,
        stdp_rule_sleep: Optional[StdpFunc] = None,
        stdp_rule_symm=None,
        n_procs=os.cpu_count(),
        callback=None,
        validation_step: int = 1,
        validation_factor: int = 10,
    ):
        assert num_batches * self.batchsize <= len(pattern)

        print(f"Number of processes: {n_procs}", flush=True)
        for step in range(num_batches):
            print(
                f"TRAINING ITERATION NO. {step} -- {datetime.now().ctime()}", flush=True
            )

            tick = time.time()
            # pick a batch:
            batch = pattern[step * self.batchsize : (step + 1) * self.batchsize]

            # every validation_step-th iteration change the sleep duration,
            # but not if validation_step == 1
            if step % validation_step == 0 and (validation_step - 1):
                self.dur_sleep *= validation_factor
                print("validation phase!")
            res = self.training_iteration(
                batch,
                stdp_rule_train,
                stdp_rule_sleep=stdp_rule_sleep,
                sal_rule=stdp_rule_symm,
                n_procs=n_procs,
            )
            if step % validation_step == 0 and (validation_step - 1):
                self.dur_sleep /= validation_factor

            quit = False
            if callback is not None:
                quit = callback(res, step)

            if quit:
                print(
                    "Recieved signal to stop training from callback function!",
                    flush=True,
                )
                break
            tock = time.time()
            print(f"Time for iteration: {tock - tick}", flush=True)

    def symmetrize(self, num_iterations, stdp_rule_symm, callback=None):
        for step in range(num_iterations):
            res = self.symmetrization_iteration(stdp_rule_symm)

            quit = False
            if callback is not None:
                quit = callback(res, step)

            if quit:
                print(
                    "Recieved signal to stop training from callback function!",
                    flush=True,
                )
                break


class NeuralSamplerFullyConnected(NeuralSampler):
    """
    A fully connected neural sampler class that extends the base NeuralSampler.

    This class implements a fully connected neural network for sampling purposes,
    with additional functionality for sleep optimization and symmetry.

    Parameters
    ----------
    init_weight : npt.NDArray
        Initial weight matrix for the neural network.
    init_bias : npt.NDArray
        Initial bias vector for the neural network.
    num_neurons : int
        Number of neurons in the network.
    sim_params : SimParams
        Simulation parameters object.
    dur_sleep : int
        Duration of the sleep phase.
    optimizer_sleep : callable
        Optimizer function for the sleep phase.
    optimizer_symm : callable, optional
        Optimizer function for symmetry, by default None.
    max_w : float, optional
        Maximum weight value, by default 2.0.
    max_b : float, optional
        Maximum bias value, by default 2.0.
    rng_seed : int, optional
        Random number generator seed, by default 424242.

    Attributes
    ----------
    dur : int
        Duration of the sleep phase.
    optimizer_sleep : callable
        Optimizer function for the sleep phase.
    optimizer_symm : callable or None
        Optimizer function for symmetry.
    los : list
        List of states for each neuron.

    Methods
    -------
    spike_rates(spikes: npt.NDArray, dur: float | int) -> np.ndarray
        Calculate spike rates for each neuron.

    """

    def __init__(
        self,
        init_weight: npt.NDArray,
        init_bias: npt.NDArray,
        target_weight: npt.NDArray,
        target_bias: npt.NDArray,
        sim_params: SimParams,
        dur_sleep: int,
        optimizer_bias: callable,
        optimizer_weight: callable,
        optimizer_symm: Optional[callable] = None,
        max_w: float = 2.0,
        max_b: float = 2.0,
        rng_seed=424242,
    ):
        """Docstring."""
        super().__init__(init_weight, init_bias, 0, sim_params, rng_seed=rng_seed)

        self.dur = dur_sleep
        self.optimizer_bias = optimizer_bias
        self.optimizer_weight = optimizer_weight
        self.optimizer_symm = optimizer_symm
        self.los = list_of_states(self.num_nrns)
        self.max_w = max_w
        self.max_b = max_b
        self.validation = False

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
        """
        Calculate spike rates for each neuron.

        Parameters
        ----------
        spikes : npt.NDArray
            Array of spike times for all neurons.
        dur : float or int
            Duration of the simulation.

        Returns
        -------
        np.ndarray
            Array of spike rates for each neuron.

        """
        rates = []
        list_of_spikes = ordered_spikes_to_list(
            spikes, list(range(1, self.num_nrns + 1))
        )
        for spks in list_of_spikes:
            rates.append(len(spks) / dur * self.t_ref)
        return np.array(rates)

    def sleep_phase(
        self, stdp_rule: StdpFunc, sal_rule: Optional[StdpFunc] = None
    ) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray, Optional[npt.NDArray]]:
        """
        Perform a single sleep phase and compute related metrics.

        This method simulates a sleep phase, calculates spike rates, applies STDP rules,
        and optionally applies SAL.

        Parameters
        ----------
        stdp_rule : StdpFunc
            The Spike-Timing Dependent Plasticity (STDP) rule to apply.
        sal_rule : StdpFunc, optional
            The SAL rule to apply, by default None.

        Returns
        -------
        Tuple[npt.NDArray, npt.NDArray, npt.NDArray, Optional[npt.NDArray]]
            A tuple containing:
            - stdp : npt.NDArray
                The STDP values computed from the sleep phase.
            - rates : npt.NDArray
                The spike rates for each neuron during the sleep phase.
            - sampled_distr : npt.NDArray
                The sampled distribution of states during the sleep phase.
            - stdp_sal : Optional[npt.NDArray]
                The SAL plasticity values, if a SAL rule was provided; otherwise, None.
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
        sal_rule: Optional[StdpFunc] = None,
    ) -> dict:
        sleep_stdp, sleep_rates, sampled_distr, stdp_sal = self.sleep_phase(
            stdp_rule=stdp_rule,
            sal_rule=sal_rule,
        )

        grad_weight = self.coact - sleep_stdp
        grad_bias = self.marginals - sleep_rates

        delta_weight = self.optimizer_weight.update(grad_weight)
        delta_bias = self.optimizer_bias.update(grad_bias)

        # update params:
        self.bias = self.bias + delta_bias
        self.weight = self.weight + delta_weight

        # optional: symmetrization with sal
        if sal_rule is not None:
            delta_sal = self.optimizer_symm(stdp_sal)
            self.weight = self.weight + delta_sal

        # impose RBM restrictions:
        self.clip_weights(self.max_w)
        self.clip_bias(self.max_b)

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
        stdp_rule_symm: Optional[StdpFunc] = None,
        callback=None,
        validation_step: int = 1,
        validation_factor: int = 10,
    ):
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
    """Stupid gardient descent

    update = lr * dL/dtheta
    """

    def __init__(self, lr):
        self.lr = lr

    def update(self, grad):
        return self.lr * grad

    def __call__(self, grad):
        return self.update(grad)
