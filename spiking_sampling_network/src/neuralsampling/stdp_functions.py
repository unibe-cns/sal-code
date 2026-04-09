"""STDP kernels, spike correlation utilities, and the STDPRuler rule container."""

from typing import Callable

import numba
import numpy as np
import numpy.typing as npt


def get_first_order_stds(
    ordered_spikes: npt.NDArray, num_neurons: int
) -> list[list[list[float]]]:
    """Return nearest-neighbor spike timing differences for all neuron pairs."""
    pre_spikes = np.full(num_neurons, -1e10)
    interspike_times = [[[] for _ in range(num_neurons)] for _ in range(num_neurons)]
    for n, spike in enumerate(ordered_spikes):
        t_post = spike[0]
        id_post = int(spike[1]) - 1
        dt_self = t_post - pre_spikes[id_post]
        for id_pre, t_pre in enumerate(pre_spikes):
            dt = t_post - t_pre
            # count only nearest pairs:
            if dt < dt_self:
                interspike_times[id_post][id_pre].append(dt)
                interspike_times[id_pre][id_post].append(-dt)
                # get isis for one neuron:
            if id_pre == id_post and dt < 1e8:
                interspike_times[id_post][id_post].append(dt)
        pre_spikes[id_post] = t_post
    return interspike_times


def get_first_order_stds_2nrn(
    ordered_spikes: npt.NDArray, num_neurons: int
) -> tuple[npt.NDArray, int]:
    """Return nearest-neighbor spike timing differences for a two-neuron network.

    Args:
        ordered_spikes: 2D array of (time, neuron_id) tuples.
        num_neurons: Must be 2.

    Returns:
        Tuple of (dts array from neuron-0 perspective, number of dropped spikes).
    """
    if num_neurons > 2:
        raise ValueError("function works only for networks with two neurons")
    pre_spikes = np.full(num_neurons, -1e10)
    interspike_times = []
    num_dropped_spikes = 0
    for n, spike in enumerate(ordered_spikes):
        t_post = spike[0]
        id_post = int(spike[1]) - 1  # Nest conuts from 1
        dt_self = t_post - pre_spikes[id_post]
        for id_pre, t_pre in enumerate(pre_spikes):
            dt = t_post - t_pre
            # count only nearest pairs:
            if dt < dt_self:
                # distinguish between causal and acausal relations,
                # seen from neuron 0
                if id_pre == 1:
                    interspike_times.append(dt)
                else:
                    interspike_times.append(-dt)
            else:
                num_dropped_spikes += 1
        pre_spikes[id_post] = t_post
    return np.asarray(interspike_times, dtype=np.float32), num_dropped_spikes


def calc_nn_stdp(
    ordered_spikes: npt.NDArray, num_neurons: int, kernel: Callable, kernel_args: tuple
) -> npt.NDArray:
    """Compute nearest-neighbor STDP correlations for all neuron pairs.

    Args:
        ordered_spikes: 2D array of (time, neuron_id) tuples.
        num_neurons: Number of neurons.
        kernel: STDP kernel function.
        kernel_args: Arguments passed to the kernel.

    Returns:
        (num_neurons, num_neurons) STDP correlation matrix.
    """
    pre_spikes = np.full(num_neurons, -1e10)
    corr_stdp = np.zeros((num_neurons, num_neurons))
    # use the identity matrix, because one has to devide corr_stdp by
    # num_in_isis (avoid division by zero)
    # num_in_isis = np.ones((num_neurons, num_neurons))  # np.eye(num_neurons)
    for i in range(len(ordered_spikes)):
        # calc correlations between neurons:
        t_post = ordered_spikes[i, 0]
        id_post = int(ordered_spikes[i, 1]) - 1
        dt_self = t_post - pre_spikes[id_post]
        for id_pre, t_pre in enumerate(pre_spikes):
            dt = t_post - t_pre
            # count only nearest pairs:
            if dt < dt_self:
                corr_stdp[id_post, id_pre] += kernel(dt, *kernel_args)
                corr_stdp[id_pre, id_post] += kernel(-dt, *kernel_args)
        pre_spikes[id_post] = t_post
    return corr_stdp


@numba.jit(nopython=True, cache=False)
def spike_corr(
    ordered_spikes: npt.NDArray,
    dt: float,
    tmax: float,
    binsize: float,
    nrn_idx: tuple[int, int] = (1, 2),
) -> float:
    """Compute the cross-correlation between two spike trains at lag dt.

    Args:
        ordered_spikes: 2D array of (time, neuron_id) tuples.
        dt: Time lag applied to the first spike train.
        tmax: Trial duration.
        binsize: Bin size for spike train discretization.
        nrn_idx: Pair of neuron IDs to correlate.

    Returns:
        Mean coincident spike count at lag dt.
    """
    spk1 = ordered_spikes[np.where(ordered_spikes[:, 1] == float(nrn_idx[0]))][:, 0]
    # shift the first spike train by dt
    spk1 += dt
    spk2 = ordered_spikes[np.where(ordered_spikes[:, 1] == float(nrn_idx[1]))][:, 0]
    # bin the spike trains:
    bins = np.arange(0, tmax + binsize, binsize)
    bins1, _ = np.histogram(spk1, bins=bins)
    bins2, _ = np.histogram(spk2, bins=bins)
    corr = np.mean(bins1 * bins2)
    return corr


@numba.jit(nopython=True)
def spike_corr_function(
    ordered_spikes: npt.NDArray,
    dts: npt.NDArray,
    tmax: float,
    binsize: float = 0.5,
    nrn_idx: tuple[int, int] = (1, 2),
) -> npt.NDArray:
    """Compute the cross-correlation function over a range of lags.

    Args:
        ordered_spikes: 2D array of (time, neuron_id) tuples.
        dts: Array of time lags to evaluate.
        tmax: Trial duration.
        binsize: Bin size for spike train discretization.
        nrn_idx: Pair of neuron IDs to correlate.

    Returns:
        Array of correlation values, one per lag in `dts`.
    """
    res = np.empty(len(dts))
    for i in numba.prange(len(dts)):
        r = spike_corr(ordered_spikes, dts[i], tmax, binsize, nrn_idx=nrn_idx)
        res[i] = r
    return res


@numba.njit(cache=False)
def exp_kernel(
    dt: float, a_plus: float, a_minus: float, tau_plus: float, tau_minus: float
) -> float:
    """Exponential STDP kernel with separate amplitudes and time constants."""
    if dt > 0.0:
        return a_plus * np.exp(-dt / tau_plus)
    elif dt < 0.0:
        return a_minus * np.exp(dt / tau_minus)
    else:
        return 0.0


@numba.njit()
def tri_kernel(
    dt: float, a_plus: float, a_minus: float, tau_plus: float, tau_minus: float
) -> float:
    """Triangular STDP kernel with separate amplitudes and time constants."""
    if dt > 0.0 and dt < tau_plus:
        return a_plus - dt * a_plus / tau_plus
    elif dt < 0.0 and dt > -tau_minus:
        return a_minus + dt * a_minus / tau_minus
    else:
        return 0.0


@numba.jit(nopython=True, cache=False)
def pairbased_stdp(
    kernel: Callable,
    kernel_args: tuple,
    ordered_spikes: npt.NDArray,
    num_neurons: int,
    num_last_spikes: int,
) -> npt.NDArray:
    """Compute pair-based STDP considering the last N spikes of each neuron.

    Args:
        kernel: STDP kernel function.
        kernel_args: Arguments passed to the kernel.
        ordered_spikes: 2D array of (time, neuron_id) tuples.
        num_neurons: Number of neurons.
        num_last_spikes: Number of past spikes to consider per neuron.

    Returns:
        (num_neurons, num_neurons) STDP matrix.
    """
    # traces = np.zeros(num_neurons)
    last_spikes = np.full((num_neurons, num_last_spikes), -np.inf)
    stdp = np.zeros((num_neurons, num_neurons))
    for i in range(len(ordered_spikes)):
        t_spk = ordered_spikes[i, 0]
        id_spk = int(ordered_spikes[i, 1]) - 1
        # add to stdp-vals:
        for id_last in range(num_neurons):
            if id_last != id_spk:
                dts = t_spk - last_spikes[id_last]
                res = 0.0
                for i in range(num_last_spikes):
                    res += kernel(dts[i], *kernel_args)
                stdp[id_spk, id_last] += res
                res = 0.0
                for i in range(num_last_spikes):
                    res += kernel(-dts[i], *kernel_args)
                stdp[id_last, id_spk] += res
        # push last_spikes one back:
        last_spikes[id_spk, :-1] = last_spikes[id_spk, 1:]
        last_spikes[id_spk, -1] = t_spk
    return stdp


@numba.jit(nopython=True, cache=False)
def noised_pairbased_stdp(
    ordered_spikes: npt.NDArray,
    kernel_func: Callable,
    a_plus: npt.NDArray,
    a_minus: npt.NDArray,
    tau_plus: npt.NDArray,
    tau_minus: npt.NDArray,
    num_neurons: int,
    num_last_spikes: int,
) -> npt.NDArray:
    """Compute pair-based STDP with per-synapse kernel parameters.

    Args:
        ordered_spikes: 2D array of (time, neuron_id) tuples.
        kernel_func: STDP kernel function.
        a_plus: (N, N) potentiation amplitudes.
        a_minus: (N, N) depression amplitudes.
        tau_plus: (N, N) potentiation time constants.
        tau_minus: (N, N) depression time constants.
        num_neurons: Number of neurons.
        num_last_spikes: Number of past spikes to consider per neuron.

    Returns:
        (num_neurons, num_neurons) STDP matrix.
    """
    # traces = np.zeros(num_neurons)
    last_spikes = np.full((num_neurons, num_last_spikes), -np.inf)
    stdp = np.zeros((num_neurons, num_neurons))
    for i in range(len(ordered_spikes)):
        t_spk = ordered_spikes[i, 0]
        id_spk = int(ordered_spikes[i, 1]) - 1
        # add to stdp-vals:
        for id_last in range(num_neurons):
            if id_last != id_spk:
                dts = t_spk - last_spikes[id_last]
                res = 0.0
                for i in range(num_last_spikes):
                    res += kernel_func(
                        dts[i],
                        a_plus[id_spk, id_last],
                        a_minus[id_spk, id_last],
                        tau_plus[id_spk, id_last],
                        tau_minus[id_spk, id_last],
                    )
                stdp[id_spk, id_last] += res
                res = 0.0
                for i in range(num_last_spikes):
                    res += kernel_func(
                        -dts[i],
                        a_plus[id_last, id_spk],
                        a_minus[id_last, id_spk],
                        tau_plus[id_last, id_spk],
                        tau_minus[id_last, id_spk],
                    )
                stdp[id_last, id_spk] += res
        # push last_spikes one back:
        last_spikes[id_spk, :-1] = last_spikes[id_spk, 1:]
        last_spikes[id_spk, -1] = t_spk
    return stdp


class STDPRuler:
    """Container for a pair-based STDP rule with per-synapse kernel parameters."""

    def __init__(
        self,
        kernel_func: Callable,
        dims: int,
        num_last_spks: int,
        a_plus: npt.ArrayLike,
        a_minus: npt.ArrayLike,
        tau_plus: npt.ArrayLike,
        tau_minus: npt.ArrayLike,
    ):
        """Initialize per-synapse kernel parameter arrays."""
        self.kernel_func = kernel_func
        self.dims = dims
        self.num_last_spks = num_last_spks

        if isinstance(a_plus, (float, int)):
            self.a_plus = np.full((dims, dims), a_plus)
        else:
            assert a_plus.shape == (dims, dims)
            self.a_plus = a_plus

        if isinstance(a_minus, (float, int)):
            self.a_minus = np.full((dims, dims), a_minus)
        else:
            assert a_minus.shape == (dims, dims)
            self.a_minus = a_minus

        if isinstance(tau_plus, (float, int)):
            self.tau_plus = np.full((dims, dims), tau_plus)
        else:
            assert tau_plus.shape == (dims, dims)
            self.tau_plus = tau_plus

        if isinstance(tau_minus, (float, int)):
            self.tau_minus = np.full((dims, dims), tau_minus)
        else:
            assert tau_minus.shape == (dims, dims)
            self.tau_minus = tau_minus

    @staticmethod
    def _copy_triu(tgt: npt.NDArray, src: npt.NDArray) -> None:
        """Copy the upper triangle of src into the upper triangle of tgt in-place."""
        tgt[np.triu_indices(tgt.shape[0], k=1)] = src[
            np.triu_indices(tgt.shape[0], k=1)
        ]

    @staticmethod
    def _copy_tril(tgt: npt.NDArray, src: npt.NDArray) -> None:
        """Copy the lower triangle of src into the lower triangle of tgt in-place."""
        tgt[np.tril_indices(tgt.shape[0], k=1)] = src[
            np.tril_indices(tgt.shape[0], k=1)
        ]

    @classmethod
    def exp_kernel(
        cls,
        dims: int,
        num_last_spks: int,
        a_plus: npt.ArrayLike,
        a_minus: npt.ArrayLike,
        tau_plus: npt.ArrayLike,
        tau_minus: npt.ArrayLike,
    ) -> "STDPRuler":
        """Construct an STDPRuler using the exponential kernel."""
        return cls(
            exp_kernel, dims, num_last_spks, a_plus, a_minus, tau_minus, tau_minus
        )

    @classmethod
    def tri_kernel(
        cls,
        dims: int,
        num_last_spks: int,
        a_plus: npt.ArrayLike,
        a_minus: npt.ArrayLike,
        tau_plus: npt.ArrayLike,
        tau_minus: npt.ArrayLike,
    ) -> "STDPRuler":
        """Construct an STDPRuler using the triangular kernel."""
        return cls(
            tri_kernel, dims, num_last_spks, a_plus, a_minus, tau_minus, tau_minus
        )

    def set_forward(
        self,
        a_plus: npt.ArrayLike | None = None,
        a_minus: npt.ArrayLike | None = None,
        tau_plus: npt.ArrayLike | None = None,
        tau_minus: npt.ArrayLike | None = None,
    ) -> None:
        """Update kernel parameters for the forward (upper triangle) direction."""
        if a_plus is not None:
            self._copy_triu(self.a_plus, a_plus)
        if a_minus is not None:
            self._copy_triu(self.a_minus, a_minus)
        if tau_plus is not None:
            self._copy_triu(self.tau_plus, tau_plus)
        if tau_minus is not None:
            self._copy_triu(self.tau_minus, tau_minus)

    def set_backward(
        self,
        a_plus: npt.ArrayLike | None = None,
        a_minus: npt.ArrayLike | None = None,
        tau_plus: npt.ArrayLike | None = None,
        tau_minus: npt.ArrayLike | None = None,
    ) -> None:
        """Update kernel parameters for the backward (lower triangle) direction."""
        if a_plus is not None:
            self._copy_tril(self.a_plus, a_plus)
        if a_minus is not None:
            self._copy_tril(self.a_minus, a_minus)
        if tau_plus is not None:
            self._copy_tril(self.tau_plus, tau_plus)
        if tau_minus is not None:
            self._copy_tril(self.tau_minus, tau_minus)

    def __call__(self, ordered_spks: npt.NDArray) -> npt.NDArray:
        """Apply the STDP rule to an ordered spike array."""
        return noised_pairbased_stdp(
            ordered_spks,
            self.kernel_func,
            self.a_plus,
            self.a_minus,
            self.tau_plus,
            self.tau_minus,
            self.dims,
            self.num_last_spks,
        )

    def noised_correlation_factors(self) -> npt.NDArray:
        """Return per-synapse theoretical correlation factors (triangular kernel only)."""
        if self.kernel_func != tri_kernel:
            raise ValueError(
                "The noised correlation factors are only defined if the STDP is triangular!"  # noqa
            )
        return 0.5 * (self.a_plus * self.tau_plus + self.a_minus * self.tau_minus)
