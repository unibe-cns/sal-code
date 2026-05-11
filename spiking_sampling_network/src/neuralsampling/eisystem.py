#!/usr/bin/env python3


import numpy as np
from scipy.special import expit

from .network import sim_poisson_neurons
from .stdp_functions import (
    calc_nn_stdp,
    exp_kernel,
    get_first_order_stds,
    spike_corr_function,
)
from .utils import spike_rate


class EISystem:
    """Excitatory-Inhibitory network: parameter storage and simulation results.

    Args:
        weights: Initial connection weights keyed by name (e.g., "e2_e1").
        bias: Initial neuron biases keyed by name (e.g., "e1").
        tref: Reference time step.
        tsyn: Synaptic time constant.
        kernel: PSP kernel function.

    Attributes:
        W: (4, 4) weight matrix.
        b: (4,) bias vector.
        rates: Firing rates per neuron (set by `calc_rates`).
        corrs: Spike correlations per connection (set by `calc_correlations`).
        stdds: Spike-time delay histograms per connection (set by `calc_stdd`).
        stdps: STDP measures for E-E synapses (set by `calc_stdp`).
    """

    NEURONS: dict[str, int] = {"e1": 0, "e2": 1, "i1": 2, "i2": 3}

    CONNECTIONS: dict[str, tuple[int, int]] = {
        "e2_e1": np.s_[1, 0],
        "e1_e2": np.s_[0, 1],
        "i1_e1": np.s_[2, 0],
        "i2_e2": np.s_[3, 1],
        "e2_i1": np.s_[1, 2],
        "e1_i2": np.s_[0, 3],
    }

    @staticmethod
    def _convert_connection_indices(np_idx: np.s_[int, int]) -> tuple[int, int]:
        """Convert numpy index slice to 1-based tuple index."""
        return (np_idx[1] + 1, np_idx[0] + 1)

    @staticmethod
    def _calc_eff_weight(
        W_ee: float, W_ie: float, W_ei: float, b_i: float, r: float
    ) -> float:
        """Compute effective excitatory weight for a connection."""
        return W_ee + W_ei * expit(W_ie * r + b_i) / r

    @staticmethod
    def _calc_eff_weight_inverse(
        W_eff: float, W_ie: float, W_ei: float, b_i: float, r: float
    ) -> float:
        """Revert effective weight back to raw excitatory weight."""
        return W_eff - W_ei * expit(W_ie * r + b_i) / r

    def __init__(
        self,
        weights: dict[str, float],
        bias: dict[str, float],
        tref: float,
        tsyn: float,
        kernel: object,
    ) -> None:
        self.tmax: float | None = None
        self.tref = tref
        self.tsyn = tsyn
        self.kernel = kernel

        n = len(self.NEURONS)
        self.W = np.zeros((n, n), dtype=float)
        for c, idx in self.CONNECTIONS.items():
            self.W[idx] = weights[c]

        self.b = np.zeros(n, dtype=float)
        for name, idx in self.NEURONS.items():
            self.b[idx] = bias[name]

        self.rates: dict[str, float] = {}
        self.corrs: dict[str, np.ndarray] = {}
        self.stdds: dict[str, np.ndarray] = {}
        self.stdps: np.zeros(2, 2)

    def _set_weight(self, idx: tuple, value: float) -> None:
        self.W[idx] = value

    def _set_bias(self, idx: int, value: float) -> None:
        self.b[idx] = value

    def _get_rate(self, name: str) -> float | None:
        return self.rates.get(name, None)

    def _get_corr(self, conn: str) -> np.ndarray | None:
        return self.corrs.get(conn, None)

    def _get_stdd(self, conn: str) -> np.ndarray | None:
        return self.stdds.get(conn, None)

    def _get_stdp(self, idx: tuple) -> float:
        return self.stdps[idx]

    def simulate(self, tmax: float) -> None:
        """Simulate spike trains for all neurons and store in `self.spks`.

        Args:
            tmax: Total simulation duration.
        """
        self.tmax = tmax
        self.spks = sim_poisson_neurons(
            self.tmax, self.kernel, self.b, self.W, self.tref, self.tsyn
        )

    def calc_rates(self) -> None:
        """Compute and store firing rates in `self.rates`. Requires `simulate()` first."""
        rates_array = spike_rate(self.spks, (1, 2, 3, 4), t_max=self.tmax) * self.tref
        self.rates = {n: r for (n, r) in zip(self.NEURONS.keys(), rates_array)}

    def calc_correlations(
        self, connections: list[str], max_dt: float = 2.0, binsize: float = 1.0
    ) -> None:
        """Compute and store spike correlations in `self.corrs`.

        Args:
            connections: Connection names to compute (e.g., ["e2_e1"]).
            max_dt: Maximum time lag in units of `tref`.
            binsize: Histogram bin size.
        """
        self.corrs = {}
        self.dts = np.arange(
            int(-max_dt * self.tref), int(max_dt * self.tref) + 1, binsize
        )
        for conn in connections:
            nrn_idx = EISystem._convert_connection_indices(EISystem.CONNECTIONS[conn])
            self.corrs[conn] = spike_corr_function(
                self.spks, self.dts, self.tmax, binsize=binsize, nrn_idx=nrn_idx
            )

    def calc_stdd(self, max_dt: float = 2.0, binsize: float = 1.0) -> None:
        """Compute and store spike-time delay distributions in `self.stdds`.

        Args:
            max_dt: Maximum delay.
            binsize: Histogram bin size.
        """
        dts = np.arange(-max_dt * self.tref, max_dt * self.tref + binsize, binsize)
        stds = get_first_order_stds(self.spks, 4)
        for conn, idx in EISystem.CONNECTIONS.items():
            self.stdds[conn], _ = np.histogram(stds[idx[0]][idx[1]], bins=dts)

    def calc_stdp(self) -> None:
        """Compute and store STDP measures for the direct E-E synapses in `self.stdps`."""
        filtered_spks = self.spks[self.spks[:, 1] < 3.0]
        stdp_kernel_args = (-1.0, 1.0, self.tref, self.tref)
        self.stdps = (
            calc_nn_stdp(filtered_spks, 2, exp_kernel, stdp_kernel_args)
            / self.tmax
            * self.tref
        )

    def transform_effective_weights_21(self, w: float) -> float:
        """Convert raw E-E weight to effective weight for connection 2→1."""
        return EISystem._calc_eff_weight(
            W_ee=w,
            W_ie=self.w_i1_e1,
            W_ei=self.w_e2_i1,
            b_i=self.b_i1,
            r=expit(self.b_e1),
        )

    def transform_effective_weights_12(self, w: float) -> float:
        """Convert raw E-E weight to effective weight for connection 1→2."""
        return EISystem._calc_eff_weight(
            W_ee=w,
            W_ie=self.w_i2_e2,
            W_ei=self.w_e1_i2,
            b_i=self.b_i2,
            r=expit(self.b_e2),
        )

    def transform_effective_weights_21_inverse(self, w: float) -> float:
        """Convert effective weight back to raw E-E weight for connection 2→1."""
        return EISystem._calc_eff_weight_inverse(
            W_eff=w,
            W_ie=self.w_i1_e1,
            W_ei=self.w_e2_i1,
            b_i=self.b_i1,
            r=expit(self.b_e1),
        )

    def transform_effective_weights_12_inverse(self, w: float) -> float:
        """Convert effective weight back to raw E-E weight for connection 1→2."""
        return EISystem._calc_eff_weight_inverse(
            W_eff=w,
            W_ie=self.w_i2_e2,
            W_ei=self.w_e1_i2,
            b_i=self.b_i2,
            r=expit(self.b_e2),
        )


# -------------------------------------------------------------------
# Property Generators (with getters/setters or read-only as required)
# -------------------------------------------------------------------

# b_* and w_*: read-write
for nname, idx in EISystem.NEURONS.items():
    setattr(
        EISystem,
        f"b_{nname}",
        property(
            lambda self, _idx=idx: self.b[_idx],
            lambda self, val, _idx=idx: self._set_bias(_idx, val),
        ),
    )

for cname, idx in EISystem.CONNECTIONS.items():
    setattr(
        EISystem,
        f"w_{cname}",
        property(
            lambda self, _idx=idx: self.W[_idx],
            lambda self, val, _idx=idx: self._set_weight(_idx, val),
        ),
    )

# rate_* and corr_*: read-write
for nname in EISystem.NEURONS.keys():
    setattr(
        EISystem,
        f"rate_{nname}",
        property(lambda self, _nname=nname: self._get_rate(_nname)),
    )

for cname in EISystem.CONNECTIONS.keys():
    setattr(
        EISystem,
        f"corr_{cname}",
        property(lambda self, _cname=cname: self._get_corr(_cname)),
    )

# effw_* and effw_inv_*: read-only
_effw_conns = {
    "e2_e1": lambda self: EISystem._calc_eff_weight(
        W_ee=self.w_e2_e1,
        W_ie=self.w_i1_e1,
        W_ei=self.w_e2_i1,
        b_i=self.b_i1,
        r=expit(self.b_e1),
    ),
    "e1_e2": lambda self: EISystem._calc_eff_weight(
        W_ee=self.w_e1_e2,
        W_ie=self.w_i2_e2,
        W_ei=self.w_e1_i2,
        b_i=self.b_i2,
        r=expit(self.b_e2),
    ),
}
for cname, func in _effw_conns.items():
    setattr(EISystem, f"effw_{cname}", property(func))

_effw_inv_conns = {
    "e2_e1": lambda self: EISystem._calc_eff_weight_inverse(
        W_eff=self.effw_e2_e1,
        W_ie=self.w_i1_e1,
        W_ei=self.w_e2_i1,
        b_i=self.b_i1,
        r=expit(self.b_e1),
    ),
    "e1_e2": lambda self: EISystem._calc_eff_weight_inverse(
        W_eff=self.effw_e1_e2,
        W_ie=self.w_i2_e2,
        W_ei=self.w_e1_i2,
        b_i=self.b_i2,
        r=expit(self.b_e2),
    ),
}
for cname, func in _effw_inv_conns.items():
    setattr(EISystem, f"effw_inv_{cname}", property(func))

# stdd_* and stdp_*: read-only
for cname in EISystem.CONNECTIONS.keys():
    setattr(
        EISystem,
        f"stdd_{cname}",
        property(lambda self, _cname=cname: self._get_stdd(_cname)),
    )

for cname in ["e1_e2", "e2_e1"]:
    setattr(
        EISystem,
        f"stdp_{cname}",
        property(
            lambda self, _cname=cname: self._get_stdp(EISystem.CONNECTIONS[_cname])
        ),
    )
