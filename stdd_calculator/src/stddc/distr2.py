#!/usr/bin/env python3

import warnings
from typing import Callable, Tuple, Union

import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.sparse.linalg import eigs

from .matrix import transition_matrix

PSPFunc = Callable[[int, int, int], float]

SalFunc = Callable[[npt.NDArray, Union[int, float], ...], npt.NDArray]


class EigenvalueError(Exception):
    """Custom exception for eigenvalue-related errors."""

    pass


class STDDMaker:
    """Docstring."""

    def __init__(
        self,
        psp: PSPFunc,
        t_max: int,
        t_ref: int,
        t_syn: int,
        w_12: float,
        w_21: float,
        b_1: float,
        b_2: float,
    ):
        """
        Initialize the STDDMaker with the given parameters.

        :param psp: A function representing the Post-Synaptic Potential.
        :type psp: PSPFunc
        :param t_max: Maximum time step.
        :type t_max: int
        :param t_ref: Refractory period.
        :type t_ref: int
        :param t_syn: Synaptic time constant.
        :type t_syn: int
        :param w_12: Synaptic weight from neuron 1 to neuron 2.
        :type w_12: float
        :param w_21: Synaptic weight from neuron 2 to neuron 1.
        :type w_21: float
        :param b_1: Bias for neuron 1.
        :type b_1: float
        :param b_2: Bias for neuron 2.
        :type b_2: float

        The PSPFunc is expected to be a callable that takes three integers and
        returns a float.
        """
        assert t_ref < t_max, "t_ref must be smaller than t_max."
        self.psp = psp
        self.t_max = t_max
        self.t_ref = t_ref
        self.t_syn = t_syn
        self.w_21 = w_12
        self.w_12 = w_21
        self.b_2 = b_1
        self.b_1 = b_2

        # construct two matrices, one that calcs the right half...
        self.T_full_right = None
        # and by exchanging the neurons the left half...
        self.T_full_left = None

        self.stdd = None

        self.times_right = np.arange(1, t_max)
        self.times_left = np.arange(-t_max + 1, 0)
        self.times = np.arange(-t_max + 1, t_max - 0)

    def create_transition_matrix(
        self, w_12: float, w_21: float, b_1: float, b_2: float
    ) -> sparse.dok_matrix:
        """
        Create a transition matrix.

        :param psp: Function representing the Post-Synaptic Potential.
        :type psp: PSPFunc
        :param t_ref: Refractory period.
        :type t_ref: int
        :param t_max: Maximum time step.
        :type t_max: int
        :param w_12: Synaptic weight from neuron 1 to neuron 2.
        :type w_12: float
        :param w_21: Synaptic weight from neuron 2 to neuron 1.
        :type w_21: float
        :param b_1: Bias for neuron 1.
        :type b_1: float
        :param b_2: Bias for neuron 2.
        :type b_2: float
        :param t_syn: Synaptic time constant.
        :type t_syn: int
        :return: A sparse transition matrix.
        :rtype: sparse.dok_matrix
        """
        T_full = sparse.dok_matrix((self.t_max**2, self.t_max**2))

        # Assuming transition_matrix is a function that yields tuples of (i, j, val)
        for i, j, val in transition_matrix(
            self.psp, self.t_ref, self.t_max, w_12, w_21, b_1, b_2, self.t_syn
        ):
            T_full[i, j] = val

        return T_full

    @staticmethod
    def calc_equilibrium_distr(
        T_full: Union[sparse.csr_matrix, sparse.csc_matrix]
    ) -> np.ndarray:
        """
        Calculate the equilibrium distribution for the given transition matrix.

        Calculates the first eigenvalue near 1.+0i and the corresponding
        eigenvector which is equal to the equilibrium distribution of the
        two neuron system.

        :param T_full: Sparse transition matrix.
        :type T_full: Union[scipy.sparse.csr_matrix, scipy.sparse.csc_matrix]
        :return: Equilibrium distribution vector.
        :rtype: numpy.ndarray
        """
        try:
            eig_val, eig_vec = eigs(T_full, k=1, sigma=1.0 + 0.0j)
        except RuntimeError:
            warnings.warn(
                "T_full seems to be singular (or numerically close to "
                "singularity). Add regularization and retry.",
                RuntimeWarning,
            )
            reg = sparse.eye(T_full.shape[0]) * 1e-10
            eig_val, eig_vec = eigs(T_full + reg, k=1, sigma=1.0 + 0.0j)

        # Find eigenvalue = 1.
        equil_id = np.where(np.isclose(eig_val, 1.0 + 0.0j))[0][0]
        if not equil_id.size == 1:
            raise EigenvalueError(
                f"Expected one eigenvalue with 1.0 + 0.0j, but found {eig_val}."
            )

        equil_distr = np.real(eig_vec[:, equil_id]).flatten()

        if not np.all(np.imag(equil_distr) == 0.0):
            raise EigenvalueError(
                f"Expected eigenvector to be real, but found {equil_distr}"
            )

        equil_distr /= np.sum(equil_distr)

        return equil_distr

    @staticmethod
    def _create_nonspiking_trans_matrix(
        T_full: Union[sparse.csr_matrix, sparse.csc_matrix], t_max: int
    ) -> sparse.csr_matrix:
        """
        Returns a copy of the transition matrix where all entries are zero that
        would let either neuron spike.
        """
        T_tilde = T_full.copy()

        # neuron 2 cannot spike
        ceta_2 = 0
        for ceta_1 in range(1, t_max):
            i = ceta_1 * t_max + ceta_2
            T_tilde[i, :] = 0.0

        # neuron 1 cannot spike
        ceta_1 = 0
        for ceta_2 in range(1, t_max):
            i = ceta_1 * t_max + ceta_2
            T_tilde[i, :] = 0.0

        T_tilde = T_tilde.tocsr()
        return T_tilde

    def _create_trans_matrix_neuron_2(
        self, T_full: Union[sparse.csr_matrix, sparse.csc_matrix]
    ) -> sparse.csr_matrix:
        """Docstring."""
        T_hat_right = sparse.dok_matrix((self.t_max, self.t_max**2))
        for ceta_1 in range(1, self.t_max):
            # if neuron 2 spikes, copy line into T_hat
            i = ceta_1 * self.t_max
            j = (ceta_1 - 1) * self.t_max + self.t_ref - 1
            k = (ceta_1 - 1) * self.t_max + self.t_max
            T_hat_right[ceta_1, j:k] = T_full[i, j:k] + T_full[0, j:k]

        T_hat_right = T_hat_right.tocsr()
        return T_hat_right

    def _calc_half_histogram(
        self, T_full: Union[sparse.csr_matrix, sparse.csc_matrix]
    ) -> np.ndarray:
        """Returns a version of T where neuron 2 will spike.

        Used to calc the RIGHT side of the dt-distribution (dt > 0)
        The dimention of the matrix is (t_max, t_max**2) --> rho = T * q, with rho
        being the desired dt-distribution.
        """
        # create the transition_matrix when no spike occurs
        T_nospike = self._create_nonspiking_trans_matrix(T_full, self.t_max)
        # create the transition_matrix when the second neuron spikes
        T_spike = self._create_trans_matrix_neuron_2(T_full)

        # calculate the equilibrium distribution
        equil_distr = self.calc_equilibrium_distr(T_full)
        # set initial distribution to states in which neuron one spikes:
        equil_distr[self.t_max :] = 0.0
        equil_distr /= np.sum(equil_distr)

        stdd = np.zeros(self.t_max)

        # Calculate the distribution of the next time step when we allow neuron 2
        # or 1 to spike
        distr_spike = T_spike.dot(equil_distr)
        stdd += distr_spike

        for t in range(self.t_max):
            # New distribution after one update where spikes are forbidden
            equil_distr = T_nospike.dot(equil_distr)
            # Calculate the distribution of the next time step when we allow
            # neuron 2 to spike
            distr_spike = T_spike.dot(equil_distr)
            stdd += distr_spike

        # cut off the last bit, because it contains all the remaining transitions
        # (i.e. an artefact comming from the existanance of a t_max)
        # stdd = stdd[:-1]
        # normalize stdd:
        stdd /= np.sum(stdd) * 2.0

        return stdd

    def calc_right(self):
        """DOCSTRING."""

        self.T_full_right = self.create_transition_matrix(
            self.w_12, self.w_21, self.b_1, self.b_2
        )

        stdd_right = self._calc_half_histogram(self.T_full_right)
        return stdd_right[1:]

    def calc_left(self):
        """DOCSTRING."""

        self.T_full_left = self.create_transition_matrix(
            self.w_21, self.w_12, self.b_2, self.b_1
        )

        stdd_left = self._calc_half_histogram(self.T_full_left)
        return stdd_left[:0:-1]

    def calc_stdd(self, fill_middle: str = "nan") -> npt.NDArray:
        """DOCSTRING."""
        ALLOWED = ["nan", "zero", "smooth"]
        assert (
            fill_middle in ALLOWED
        ), f"optional argument 'fill_middle' must be from {ALLOWED}"

        stdd_right = self.calc_right()
        stdd_left = self.calc_left()

        if fill_middle == "nan":
            fill_value = np.array([np.nan])
        elif fill_middle == "zero":
            fill_value = np.array([0.0])
        else:
            a = (stdd_left[-1] + stdd_right[0]) * 0.5
            fill_value = np.array([a])

        self.stdd = np.concatenate([stdd_left, fill_value, stdd_right])

        return np.copy(self.stdd)

    def calc_sal(self, func: SalFunc, *args: Union[int, float]) -> Tuple[float, float]:
        """DOCSTRING."""
        kernel_vals = func(self.times, *args)

        if self.stdd is None:
            stdd = self.calc_stdd(fill_middle="zero")
        else:
            stdd = np.copy(self.stdd)
            stdd[self.t_max - 1] = 0.0

        sal_12 = np.sum(stdd * kernel_vals)
        sal_21 = np.sum(stdd * kernel_vals[::-1])

        return sal_12, sal_21
