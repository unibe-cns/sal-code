from itertools import product
from pathlib import Path
from typing import Callable, List, Optional, Tuple, TypeAlias

import matplotlib.pyplot as plt
import numba
import numpy as np
import numpy.typing as npt
import yaml

# declare my own types here
StrPath: TypeAlias = Path | str
ParamDict: TypeAlias = dict  # TODO


def logistic(x, x0, alpha):
    """
    logistic function, shifted hy x0 and scaled by alpha
    """
    return 1.0 / (1.0 + np.exp(-(x - x0) / alpha))


@numba.jit(nopython=True)
def get_states_from_spikes(
    number_of_neurons: int, spikes, taurefs, dt: float, duration: float = 0.0
):
    """
    Return a list of states, sampled at multiples of dt, for the neurons produced by
    spikes.

    Input:
    number_of_neurons   int     size of the resulting state
    spikes              ndarray 2d array of spike (time, id)-tuple
    taurefs             ndarray tauref values to be used for all neurons, must be
                                number_of_neurons long

    If duration is not explicitly specified it will be inferred from the last spike

    The state of neuron i at time t is 1 iff there was a spike within (t-tauref[i], t)
    We assume that the provided taurefs are correct, i.e. there are no two spikes of
    a single neuron such that t[n+1]-t[n] < tauref. If that assumption is violated we
    raise a warning.

    Taken from hxsampling :)
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


def number_to_state(number, nNeurons):
    """
    Returns the state corresponding to a given index

    """
    if number >= 2**nNeurons:
        raise ValueError(
            f"{number} is to larger to be a valid state for {nNeurons} neurons."
        )
    s = tuple(int(s) for s in bin(int(number))[2:])
    state = np.array((nNeurons - len(s)) * (0,) + s)

    return state


def bm_to_probs(W, b, force=False):
    """Calculate the the probability distribution, the marginals of the single
    neurons and the pariwise joint distributions for an abstract Boltzmann
    machine analytically
    Keywords:
        -- W: connection matrix
        -- b: bias vector
        -- force: force the calculation for more then 15 neurons
    Returns:
        -- probs: probability distribution
        -- statesArr: array of states corresponding to probs
        -- C: pairwise joint, the diagonal contains the marginals
    """

    # reject working on more than 15 neurons unless forced
    N = len(b)
    if N > 15 and not force:
        raise ValueError(
            "The function takes too long for more than 15\
          neurons. Received {}. To calculate anyway, use\
          the force option.".format(
                N
            )
        )
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


def vis_distr_from_rbm(n_vis, n_hid, weights, bias):
    """Calculate the joint distribution of the visibles in a RBM

    Arguments:
        n_vis (int): number of visible neurons
        n_hid (int): number of hidden neurons
        weights (np.ndarray): weight matrix of RBM
        bias (np.ndarray): bias vector of RBM

    Return:
        all_vis_joint_prob (np.ndarray): joint probability of the visibles
        all_vis_states: List of tuples with all visible states.
    """
    n_tot = n_vis + n_hid
    if n_tot > 20:
        print(f"RBM is very large ({n_tot} neurons)! Computing time my be too long!")

    all_vis_states = list(product(range(2), repeat=n_vis))
    all_hid_states = list(product(range(2), repeat=n_hid))

    all_vis_joint_prob = np.zeros(2**n_vis)
    part_func = 0.0  # Normalization factor
    # iterate over all combinations of hidden states
    for i, vis_state in enumerate(all_vis_states):
        prob = 0.0
        # sum over all combinations of hidden states
        for hid_state in all_hid_states:
            state = np.array(vis_state + hid_state)
            energy = -0.5 * np.dot(state, np.dot(weights, state)) + np.dot(bias, state)
            prob += np.exp(-energy)
        part_func += prob
        all_vis_joint_prob[i] = prob

    # Normalize
    all_vis_joint_prob /= part_func

    return all_vis_joint_prob, all_vis_states


def list_of_states(num_neurons):
    """
    create a list of all possible states
    """
    return list(product(range(2), repeat=num_neurons))


def distr_from_states(sampledStates, states):
    """
    Calculate the probabilities of the joint states, from sampled states.

    The implemented method:
    In the code we assign to each state an integer id. We do this
    by interpreting the neural state as a binary number.
    For example: The state [1,1,0,0,1] is assigned the number
    2^4 * 1 + 2^3 * 1 + 2^2 * 0 + 2^1 * 0 + 2^0 * 1 = 25
    We do this assingement with both the states found in <sampledStates>
    and in <states>. Finally, we count how often the IDs in <states<
    appear in <sampledStates>.

    Args:
        -- sampledStates: matrix of all sampled states
                          the states are in rows
        -- states: list of states, the returned probabilities will
                   correspond to the order of these states
    Return:
        -- probs: array of probabilities
    """

    # cast the input to integeres be able to use integer operations
    # to increase the execution speed
    sampledStates = sampledStates.astype(int)
    states = np.array(states).astype(int)

    # Turn the sampled states into a list of state IDs
    # the << is the bitshift operator which can be used for fast calculation
    # of the state IDs. E.g. "x << y" evaluates to "x * 2**y"
    isampledStates = (
        sampledStates
        << np.tile(
            np.arange(sampledStates.shape[1] - 1, -1, -1), (sampledStates.shape[0], 1)
        )
    ).sum(axis=1)

    # Turn the searched samples into a list of state IDs
    istates = (
        states << np.tile(np.arange(states.shape[1] - 1, -1, -1), (states.shape[0], 1))
    ).sum(axis=1)

    # Count the IDs of interest in the sampled IDs
    counts = np.bincount(isampledStates, minlength=len(states))

    # Nomalize the frequencies to obtain probabilities and ensure that the
    # order follows the one provided in states
    prob = counts[istates] / counts.sum()

    return prob


@numba.njit()
def calc_coact_from_states(states):
    """Calc the coactivation and the marginals from binary states"""
    num_samples, num_neurons = states.shape
    coact = np.zeros((num_neurons, num_neurons))
    for i in range(num_samples):
        coact += np.outer(states[i, :], states[i, :])
    coact /= num_samples
    marginals = np.diag(coact).copy()
    np.fill_diagonal(coact, 0.0)
    return coact, marginals


def calc_dkl(p, q):
    """
    Kullback-Leibler divergence
    """
    return np.sum(p * np.log(p / q))


def ordered_spikes_to_list(ordered_spikes, neuron_ids):
    """Turn a ordered spikes array into a list of spike arrays

    Ordered spikes contains a two-dim numpy array, where the first entry stores
    the spike time and the second entry the respective neuron id.
    Returns a list of arrays for each neuron with the spike times (typically
    used when plotting spike rasterplots).
    neuron_ids is a list with the respective neuron ids.
    """
    spike_list = []
    for id in neuron_ids:
        # a quick measurement revealed that a[np.where(a) == x] is a little bit
        # faster than a[a == x]
        spks = ordered_spikes[np.where(ordered_spikes[:, 1] == float(id))][:, 0]
        spike_list.append(spks)
    return spike_list


def spike_rate(ordered_spikes, nrn_ids, t_max=None, timeunit="ms"):
    """Calculate the mean spike rates from an ordered spike train.

    Asumes that the spike times are given in milli seconds. Also returns rates in
    1/ms. If t_max is None, take the last spike as an estimate for the trial duration.
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


def prettyprint_distribution(list_of_states, *distributions):
    """pretty print the distribution

    one hast to pass a list of states array and one or many distributions
    """
    for elements in zip(list_of_states, *distributions):
        print(*elements)


def rasterplot(spike_arrays, neuron_ids, colors=None, zoom=None, savefig=None):
    """Plot a spike rasterplot

    Plots a raster plot for spike trains.
    Takes a list (must always be a list) of ordered numpy spike arrays in HX style
    and a list of lists with the corresponding neuron ids.
    Additionall, one can give list of colors --> each orderd spike array
    (=group of neurons) gets its own color.
    Also, one can specify a zoom in form of [t_min, t_max].
    Save the figure under the name specified in savefig.
    """
    assert len(spike_arrays) == len(neuron_ids)
    if colors is not None:
        assert len(colors) == len(spike_arrays)

    # turn ordered spikes into lists of spikes:
    spike_list = []
    neuron_ids_list = []
    cs = []

    # if no color is specified, use the standard color for all chucks:
    if colors is None:
        colors = ["C0"] * len(spike_arrays)
    for spk_arr, nrn_ids, c in zip(spike_arrays, neuron_ids, colors):
        # zoom into the time line:
        if zoom is not None:
            zoom_ids = np.where((spk_arr[:, 0] > zoom[0]) & (spk_arr[:, 0] < zoom[1]))
            spk_arr = spk_arr[zoom_ids]
        intermediate_spikelist = ordered_spikes_to_list(spk_arr, nrn_ids)
        for spks, id in zip(intermediate_spikelist, nrn_ids):
            spike_list.append(spks)
            neuron_ids_list.append(id)
            cs.append(c)

    fig, ax = plt.subplots()
    ax.eventplot(spike_list, lineoffsets=neuron_ids_list, colors=cs)
    ax.set_xlabel(r"$t_{spikes}$ [ms]")
    ax.set_ylabel("neuron id")

    if savefig is not None:
        fig.savefig(savefig, dpi=300)
    else:
        plt.show()

    return fig, ax


def plot_distr(
    ax: plt.Axes,
    distrs: List[np.ndarray],
    los: List[str],
    labels: Optional[List[str]] = None,
) -> plt.Axes:
    """
    Plot multiple distributions as grouped bar charts on a logarithmic scale.

    This function creates a grouped bar chart for multiple distributions,
    with each distribution represented by a different color. The y-axis is
    set to a logarithmic scale.

    Parameters
    ----------
    ax : plt.Axes
        The matplotlib Axes object on which to draw the plot.
    distrs : List[np.ndarray]
        A list of 1D numpy arrays, each representing a distribution to be plotted.
    los : List[str]
        A list of strings representing the labels for each state (x-axis labels).
    labels : Optional[List[str]], default=None
        A list of labels for each distribution. If None, no labels are used.

    Returns
    -------
    plt.Axes
        The matplotlib Axes object with the plotted distributions.

    Notes
    -----
    - The function assumes all distributions have the same number of states.
    - The bars for different distributions are grouped together for each state.
    - The y-axis is set to a logarithmic scale to better visualize variations
      in probability across orders of magnitude."""
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


def load_paramfile(path: StrPath) -> ParamDict:
    """DOCSTRING."""
    with open(path, "r") as f:
        params = yaml.safe_load(f)
        return params


def create_training_pattern(
    num: int,
    distr: npt.NDArray,
    pattern_seed: Optional[int] = None,
) -> tuple[npt.NDArray, npt.NDArray]:
    """DOCSTRING."""

    dim = int(np.log2(len(distr)))
    los = list_of_states(dim)

    prng = np.random.default_rng(pattern_seed)
    pattern = prng.choice(los, size=num, p=distr)

    return pattern


def create_distr_from_uniform(
    dim: int, distr_seed: Optional[int] = None
) -> npt.NDArray:
    """DOSCTRING."""
    drng = np.random.default_rng(distr_seed)
    distr = drng.random(2**dim)
    distr = distr / np.sum(distr)

    return distr


def restrict(w: npt.NDArray, dim_vis: int) -> None:
    """DOCSTRING."""
    w[:dim_vis, :dim_vis] = 0.0
    w[dim_vis:, dim_vis:] = 0.0


def create_rbm_weights(dim_vis: int, dim_hid: int, w: npt.NDArray) -> npt.NDArray:
    """DOCSTRING."""
    dim = dim_vis + dim_hid
    for i in range(dim):
        for j in range(i + 1, dim):
            w[j, i] = w[i, j]
    restrict(w, dim_vis)
    return w


def check_in(array_1d: npt.NDArray, array_2d: npt.NDArray, axis: int = 1) -> bool:
    """Check if array_1d is contained in array_2d along axis."""
    return np.any(
        np.apply_along_axis(
            lambda row: np.array_equal(row, array_1d), axis=axis, arr=array_2d
        )
    )


def check_training_data(
    distr: npt.NDArray, pattern: npt.NDArray
) -> tuple[float, npt.NDArray]:
    """
    Calculate the Kullback-Leibler divergence between target distribution and samples.

    This function computes the Kullback-Leibler divergence (DKL) between a target
    distribution and the distribution of sampled patterns. It also handles cases
    where certain states might not appear in the sampled patterns.

    Parameters
    ----------
    distr : npt.NDArray
        The target distribution. Should be a 1D array representing probabilities
        for all possible states.
    pattern : npt.NDArray
        The sampled patterns. Should be a 2D array where each row represents a
        sampled state.

    Returns
    -------
    tuple[float, npt.NDArray]
        A tuple containing:
        - float: The calculated Kullback-Leibler divergence.
        - npt.NDArray: The empirical distribution of the sampled patterns.
    """
    dims = int(np.log2(len(distr)))
    los = list_of_states(dims)
    uniques, counts = np.unique(pattern, return_counts=True, axis=0)
    # check for non_existing:
    for i, pat in enumerate(los):
        if not check_in(np.array(pat), uniques):
            counts = np.insert(counts, i, 0.0)
    pat_distr = counts / len(pattern)
    dkl = calc_dkl(distr, pat_distr)
    return dkl, pat_distr


def draw_trunc_distr(
    generator: Callable, high: float, low: float, size: Tuple[int], kwargs: dict = {}
):
    """
    Draw random numbers from a truncated distribution within specified boundaries.

    This function generates random numbers using the provided generator function,
    ensuring all values fall within the specified range [low, high]. If any generated
    numbers are outside this range, they are redrawn until all values are valid.

    Parameters
    ----------
    generator : Callable
        A function that generates random numbers. It should accept a `size` parameter
        and any additional keyword arguments provided in `kwargs`.
    high : float
        The upper boundary of the acceptable range.
    low : float
        The lower boundary of the acceptable range.
    size : Tuple[int]
        The desired shape of the output array.
    kwargs : dict, optional
        Additional keyword arguments to pass to the generator function (default is {}).

    Returns
    -------
    np.ndarray
        An array of random numbers with shape `size`, all within the range [low, high]."
    """

    res = generator(size=np.prod(size), **kwargs)
    invalid = (res < low) | (res > high)
    invalid_count = np.sum(invalid)

    while invalid_count > 0:
        res[invalid] = generator(size=invalid_count, **kwargs)
        invalid = (res < low) | (res > high)
        invalid_count = np.sum(invalid)

    return res.reshape(size)


def copy_triu(arr: npt.NDArray) -> npt.NDArray:
    """Copy the upper triangular part of a matrix to its lower triangular part.

    This function creates a symmetric matrix by copying the upper triangular
    elements (excluding the main diagonal) to the corresponding lower triangular
    positions.

    Parameters
    ----------
    arr : npt.NDArray
        Input square matrix.

    Returns
    -------
    npt.NDArray
        A new matrix with the upper triangular part copied to the lower triangular part.

    Notes
    -----
    The function assumes that `arr` is a square matrix.
    The main diagonal remains unchanged.
    """
    res = np.copy(arr)
    res[np.tril_indices(arr.shape[0])] = 0.0
    return res + res.T


def copy_tril(arr: npt.NDArray) -> npt.NDArray:
    """Copy the lower triangular part of a matrix to its upper triangular part.

    This function creates a symmetric matrix by copying the lower triangular
    elements (excluding the main diagonal) to the corresponding upper triangular
    positions.

    Parameters
    ----------
    arr : npt.NDArray
        Input square matrix.

    Returns
    -------
    npt.NDArray
        A new matrix with the lower triangular part copied to the upper triangular part.

    Notes
    -----
    The function assumes that `arr` is a square matrix.
    The main diagonal remains unchanged.
    """
    res = np.copy(arr)
    res[np.triu_indices(arr.shape[0])] = 0.0
    return res + res.T
