from itertools import product
from pathlib import Path
from typing import Callable, TypeAlias

import matplotlib.pyplot as plt
import numba
import numpy as np
import numpy.typing as npt
import yaml
from scipy.special import xlogy

# declare my own types here
StrPath: TypeAlias = Path | str
ParamDict: TypeAlias = dict  # TODO


@numba.jit(nopython=True)
def get_states_from_spikes(
    number_of_neurons: int,
    spikes: npt.NDArray,
    taurefs: npt.NDArray,
    dt: float,
    duration: float = 0.0,
) -> npt.NDArray:
    """Sample neuron states at multiples of dt from an ordered spike array.

    Args:
        number_of_neurons: Number of neurons.
        spikes: 2D array of (time, neuron_id) spike tuples.
        taurefs: Refractory period per neuron.
        dt: Sampling interval.
        duration: Simulation duration; inferred from last spike if 0.

    Returns:
        2D state array of shape (num_timesteps, number_of_neurons).
    """
    state = np.zeros(number_of_neurons)
    # buffer to keep the outstanding offspikes around, should not exceed size
    # number_of_neurons
    offspikes = []

    if len(spikes) > 0:
        if duration == 0.0:
            duration = spikes[-1][0] + taurefs[int(spikes[-1][1])]

        times = np.arange(0.0, duration, dt)
        states = np.zeros((len(times), number_of_neurons))

        for nsample, sampletime in enumerate(times):
            # deal with all off spikes that occured, we drop the ones that we
            # already dealt with, len(offspikes)<number_of_neurons
            # Note: This assumes that we use a correct tauref and min(isi[n])
            #       is always larger than tauref[n]
            while True:
                if len(offspikes) > 0:
                    if offspikes[0][0] < sampletime:
                        state[int(offspikes[0][1]) - 1] = 0
                        offspikes.pop(0)
                    else:
                        break
                else:
                    break

            # Handle all spikes since the last timestep. Create the new state
            # Note: We might want to keep the spikes around in case we want to
            #       track our progress
            while spikes.size != 0 and spikes[0][0] < sampletime:
                extr_spike, spikes = spikes[0], spikes[1:]
                spiketime = extr_spike[0]
                neuronid = extr_spike[1]
                state[int(neuronid) - 1] = 1
                for i, (st, nid) in enumerate(offspikes):
                    if nid == neuronid:
                        # Multiple spikes within one tauref detected dropping
                        # the additional one
                        offspikes.pop(i)
                        # This break is a performance optimisation and only
                        # correct if there is at most one offspike added
                        # between two successive runs
                        break
                offspikes.append((spiketime + taurefs[int(neuronid) - 1], neuronid))

            states[nsample, :] = state

        return states
    else:
        return np.zeros((2, number_of_neurons))


def number_to_state(number: int, n_neurons: int) -> npt.NDArray:
    """Convert an integer index to its binary neuron state vector."""
    if number >= 2**n_neurons:
        raise ValueError(
            f"{number} is to larger to be a valid state for {n_neurons} neurons."
        )
    s = tuple(int(s) for s in bin(int(number))[2:])
    state = np.array((n_neurons - len(s)) * (0,) + s)

    return state


def bm_to_probs(W: npt.NDArray, b: npt.NDArray, force: bool = False) -> list:
    """Compute the Boltzmann machine probability distribution analytically.

    Args:
        W: Weight matrix.
        b: Bias vector.
        force: Allow computation for >15 neurons.

    Returns:
        [probs, states, coactivation] — probability array, list of states,
        pairwise joint matrix (diagonal = marginals).
    """

    # reject working on more than 15 neurons unless forced
    N = len(b)
    if N > 15 and not force:
        raise ValueError("The function takes too long for more than 15\
          neurons. Received {}. To calculate anyway, use\
          the force option.".format(N))
    values = np.zeros(2**N)
    coactivation = np.zeros((N, N))

    # Loop over the possible states
    for i in range(2**N):
        state = number_to_state(i, N)
        energy = -1.0 * (0.5 * np.dot(state, np.dot(W, state)) + np.dot(b, state))
        value = np.exp(-1.0 * energy)
        values[i] = value
        coactivation += value * np.outer(state, state)

    # Normalize
    Z = np.sum(values)
    probs = values / Z
    coactivation = coactivation / Z

    # Array of the states
    # The format has to be compatible with the logic in
    # target.Targetdistribution

    statesArr = [number_to_state(i, N) for i in np.arange(2**N)]

    return [probs, statesArr, coactivation]


def list_of_states(num_neurons: int) -> list[tuple[int, ...]]:
    """
    create a list of all possible states
    """
    return list(product(range(2), repeat=num_neurons))


def distr_from_states(sampled_states: npt.NDArray, states: list) -> npt.NDArray:
    """Estimate the joint state distribution from sampled states.

    Args:
        sampled_states: Matrix of sampled states (states in rows).
        states: Reference list of states to compute probabilities for.

    Returns:
        Probability array aligned to the order of `states`.
    """

    # cast the input to integeres be able to use integer operations
    # to increase the execution speed
    sampled_states = sampled_states.astype(int)
    states = np.array(states).astype(int)

    # Turn the sampled states into a list of state IDs
    # the << is the bitshift operator which can be used for fast calculation
    # of the state IDs. E.g. "x << y" evaluates to "x * 2**y"
    isampled_states = (
        sampled_states
        << np.tile(
            np.arange(sampled_states.shape[1] - 1, -1, -1), (sampled_states.shape[0], 1)
        )
    ).sum(axis=1)

    # Turn the searched samples into a list of state IDs
    istates = (
        states << np.tile(np.arange(states.shape[1] - 1, -1, -1), (states.shape[0], 1))
    ).sum(axis=1)

    # Count the IDs of interest in the sampled IDs
    counts = np.bincount(isampled_states, minlength=len(states))

    # Nomalize the frequencies to obtain probabilities and ensure that the
    # order follows the one provided in states
    prob = counts[istates] / counts.sum()

    return prob


def calc_dkl(p: npt.NDArray, q: npt.NDArray) -> float:
    """
    Kullback-Leibler divergence
    """
    return float(np.sum(xlogy(p, p) - xlogy(p, q)))


def ordered_spikes_to_list(
    ordered_spikes: npt.NDArray, neuron_ids: list[int]
) -> list[npt.NDArray]:
    """Split an ordered spike array into per-neuron spike time lists.

    Args:
        ordered_spikes: 2D array of (time, neuron_id) tuples.
        neuron_ids: List of neuron IDs to extract.

    Returns:
        List of 1D spike-time arrays, one per neuron in `neuron_ids`.
    """
    spike_list = []
    for id in neuron_ids:
        # a quick measurement revealed that a[np.where(a) == x] is a little bit
        # faster than a[a == x]
        spks = ordered_spikes[np.where(ordered_spikes[:, 1] == float(id))][:, 0]
        spike_list.append(spks)
    return spike_list


def spike_rate(
    ordered_spikes: npt.NDArray,
    nrn_ids: list[int] | tuple[int, ...],
    t_max: float | None = None,
    timeunit: str = "ms",
) -> npt.NDArray:
    """Compute mean spike rates from an ordered spike train.

    Args:
        ordered_spikes: 2D array of (time, neuron_id) tuples.
        nrn_ids: Neuron IDs to compute rates for.
        t_max: Trial duration; inferred from last spike if None.
        timeunit: 'ms' or 's'.

    Returns:
        Array of mean spike rates, one per neuron in `nrn_ids`.
    """
    if t_max is None:
        t_max = ordered_spikes[-1, 0]

    assert timeunit in ["ms", "s"]

    conversion = {"ms": 1.0, "s": 1000.0}

    los = ordered_spikes_to_list(ordered_spikes, nrn_ids)
    rates = []
    for spks in los:
        rates.append(len(spks) / t_max * conversion[timeunit])

    return np.array(rates)


def plot_distr(
    ax: plt.Axes,
    distrs: list[npt.NDArray],
    los: list[str],
    labels: list[str] | None = None,
) -> plt.Axes:
    """Plot multiple distributions as grouped bar charts on a log scale.

    Args:
        ax: Matplotlib axes to draw on.
        distrs: List of probability arrays to plot.
        los: State labels for x-axis.
        labels: Legend labels per distribution.

    Returns:
        The axes with the plot.
    """
    num_distrs = len(distrs)
    num_states = len(distrs[0])
    xs_distrs = np.arange(num_states)
    width = 0.8 / num_distrs
    displacement = -(num_distrs - 1.0) / 2.0 * width

    x_ticklabels = [str(state) for state in los]

    # displacement = 0.0
    ax.set_yscale("log")

    labels = [""] * len(distrs) if labels is None else labels

    for i in range(num_distrs):
        print(displacement)
        x_pos = xs_distrs + displacement
        ax.bar(x_pos, distrs[i], width, label=labels[i])
        ax.set_xticks(xs_distrs)
        ax.set_xticklabels(x_ticklabels, rotation="vertical", fontsize=8)
        displacement += width

    ax.set_xlabel("States")
    ax.set_ylabel("p(z)")
    plt.tight_layout()
    if labels is not None:
        ax.legend()

    return ax


def load_paramfile(
    path: StrPath,
) -> ParamDict:
    """Load a YAML parameter file and return its contents as a dict."""
    with open(path, "r") as f:
        params = yaml.safe_load(f)
        return params


def draw_trunc_distr(
    generator: Callable,
    high: float,
    low: float,
    size: tuple[int, ...],
    kwargs: dict = {},
) -> npt.NDArray:
    """Draw random samples from a distribution truncated to [low, high].

    Resamples values outside the range until all are valid.

    Args:
        generator: Random number generator callable (e.g., rng.normal).
        high: Upper bound.
        low: Lower bound.
        size: Shape of the output array.
        kwargs: Extra keyword arguments forwarded to `generator`.

    Returns:
        Array of shape `size` with all values in [low, high].
    """

    res = generator(size=np.prod(size), **kwargs)
    invalid = (res < low) | (res > high)
    invalid_count = np.sum(invalid)

    while invalid_count > 0:
        res[invalid] = generator(size=invalid_count, **kwargs)
        invalid = (res < low) | (res > high)
        invalid_count = np.sum(invalid)

    return res.reshape(size)


def copy_triu(
    arr: npt.NDArray,
) -> npt.NDArray:
    """Symmetrize a matrix by copying the upper triangle to the lower.

    Args:
        arr: Square input matrix.

    Returns:
        New symmetric matrix with upper triangle values mirrored.
    """
    res = np.copy(arr)
    res[np.tril_indices(arr.shape[0])] = 0.0
    return res + res.T


def copy_tril(arr: npt.NDArray) -> npt.NDArray:
    """Symmetrize a matrix by copying the lower triangle to the upper.

    Args:
        arr: Square input matrix.

    Returns:
        New symmetric matrix with lower triangle values mirrored.
    """
    res = np.copy(arr)
    res[np.triu_indices(arr.shape[0])] = 0.0
    return res + res.T
