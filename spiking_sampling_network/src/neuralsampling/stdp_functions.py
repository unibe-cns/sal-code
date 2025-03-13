"""DOCSTRIG."""

from typing import Optional

import numba
import numpy as np
import numpy.typing as npt


def get_first_order_stds(ordered_spikes, num_neurons):
    """Obtain the first order spikes timing differences from a list of ordered
    spikes
    """
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


def get_first_order_stds_2nrn(ordered_spikes, num_neurons):
    """Obtain the dts between two neurons from a list of ordered spikes

    Uses HX's next neighbor counting scheme. Works only if network consists of
    two neurons. Uses less memory compared to get_interspike_times. Returns a
    1-dim np.ndarray with the dts between the two neurons from the perspective
    of neuron 1.

    Input:
        ordered_spikes  (list or np.ndarray) ordered spikes return by sbs
        num_neurons     (int)
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


# @jit(nopython=True)
def calc_nn_stdp(ordered_spikes, num_neurons, kernel, kernel_args):
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
def spike_corr(ordered_spikes, dt, tmax, binsize, nrn_idx=(1, 2)):
    spk1 = ordered_spikes[np.where(ordered_spikes[:, 1] == float(nrn_idx[0]))][:, 0]
    # shift the first spike train by dt
    spk1 += dt
    spk2 = ordered_spikes[np.where(ordered_spikes[:, 1] == float(nrn_idx[1]))][:, 0]
    # bin the spike trains:
    bins = np.arange(0, tmax + binsize, binsize)
    bins1, _ = np.histogram(spk1, bins=bins)
    bins2, _ = np.histogram(spk2, bins=bins)
    cov = np.cov(np.vstack((bins1, bins2)))
    return cov[0, 1]


@numba.jit(nopython=True)
def spike_corr_function(ordered_spikes, dts, tmax, binsize=0.5, nrn_idx=(1, 2)):
    res = np.empty_like(dts)
    for i in numba.prange(len(dts)):
        res[i] = spike_corr(ordered_spikes, dts[i], tmax, binsize, nrn_idx=nrn_idx)
    return res


@numba.njit(cache=False)
def exp_kernel(dt, a_plus, a_minus, tau_plus, tau_minus):
    if dt > 0.0:
        return a_plus * np.exp(-dt / tau_plus)
    elif dt < 0.0:
        return a_minus * np.exp(dt / tau_minus)
    else:
        return 0.0


@numba.njit()
def tri_kernel(dt, a_plus, a_minus, tau_plus, tau_minus):
    if dt > 0.0 and dt < tau_plus:
        return a_plus - dt * a_plus / tau_plus
    elif dt < 0.0 and dt > -tau_minus:
        return a_minus + dt * a_minus / tau_minus
    else:
        return 0.0


@numba.jit(nopython=True, cache=False)
def pairbased_stdp(kernel, kernel_args, ordered_spikes, num_neurons, num_last_spikes):
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
    kernel_func: callable,
    a_plus: npt.NDArray,
    a_minus: npt.NDArray,
    tau_plus: npt.NDArray,
    tau_minus: npt.NDArray,
    num_neurons: int,
    num_last_spikes: int,
) -> npt.NDArray:
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
    """DOCSTRING."""

    def __init__(
        self,
        kernel_func: callable,
        dims: int,
        num_last_spks: int,
        a_plus: npt.ArrayLike,
        a_minus: npt.ArrayLike,
        tau_plus: npt.ArrayLike,
        tau_minus: npt.ArrayLike,
    ):
        """DOCSTRING."""
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
        """DOCSTRING."""
        tgt[np.triu_indices(tgt.shape[0], k=1)] = src[
            np.triu_indices(tgt.shape[0], k=1)
        ]

    @staticmethod
    def _copy_tril(tgt: npt.NDArray, src: npt.NDArray) -> None:
        """DOCSTRING."""
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
    ):
        """Docstring."""
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
    ):
        """Docstring."""
        return cls(
            tri_kernel, dims, num_last_spks, a_plus, a_minus, tau_minus, tau_minus
        )

    def set_forward(
        self,
        a_plus: Optional[npt.ArrayLike] = None,
        a_minus: Optional[npt.ArrayLike] = None,
        tau_plus: Optional[npt.ArrayLike] = None,
        tau_minus: Optional[npt.ArrayLike] = None,
    ) -> None:
        """DOCSTRING."""
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
        a_plus: Optional[npt.ArrayLike] = None,
        a_minus: Optional[npt.ArrayLike] = None,
        tau_plus: Optional[npt.ArrayLike] = None,
        tau_minus: Optional[npt.ArrayLike] = None,
    ) -> None:
        """DOCSTRING."""
        if a_plus is not None:
            self._copy_tril(self.a_plus, a_plus)
        if a_minus is not None:
            self._copy_tril(self.a_minus, a_minus)
        if tau_plus is not None:
            self._copy_tril(self.tau_plus, tau_plus)
        if tau_minus is not None:
            self._copy_tril(self.tau_minus, tau_minus)

    def __call__(self, ordered_spks: npt.NDArray) -> npt.NDArray:
        """DOCSTRING."""
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
