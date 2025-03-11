"""Calculate the STDP-Streamplot analytically

Calc the STDP-Streamplot of a two neuron buesing network with arbitrary PSP
shape analytically.

Detailed description tba
"""

import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse as sm
from scipy.sparse.linalg import eigs

from .matrix import transition_matrix

logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%d/%m/%Y %H:%M:%S",
)


def rect_PSP(zeta, tau_ref, tau_syn):
    """rectangular psp shape"""
    return np.heaviside(-zeta + tau_ref, 0.0) * np.heaviside(zeta, 1.0)


def exp_PSP(zeta, tau_ref, tau_syn):
    """exponential psp shape"""
    a = (tau_ref / tau_syn) / (1 - np.exp(-tau_ref / tau_syn))
    return a * np.exp(-zeta / tau_syn)


def cutoff_PSP(zeta, tau_ref, tau_syn):
    if zeta <= tau_ref:
        a = (tau_ref / tau_syn) / (1 - np.exp(-tau_ref / tau_syn))
        return a * np.exp(-zeta / tau_syn)
    else:
        return 0.0


def tail_PSP(zeta, tau_ref, tau_syn):
    if zeta <= tau_ref:
        return 1.0
    else:
        a = (tau_ref / tau_syn) / (1 - np.exp(-tau_ref / tau_syn))
        return a * np.exp(-zeta / tau_syn)


def alpha_PSP(zeta, tau_ref, tau_syn):
    if zeta > 0.0:
        return tau_ref / tau_syn**2 * zeta * np.exp(-zeta / tau_syn)
    else:
        return 0.0


def create_transition_matrix(PSP, tau_ref, t_max, w_12, w_21, b_1, b_2, tau_syn):
    """Create a sparse transition matrix"""
    T_sparse = sm.dok_matrix((t_max**2, t_max**2))

    for i, j, res in transition_matrix(
        PSP, tau_ref, t_max, w_12, w_21, b_1, b_2, tau_syn
    ):
        T_sparse[i, j] = res
    logging.info(
        "Size of sparse matrix in MB {}".format(sys.getsizeof(T_sparse) / 1.0e6)
    )

    return T_sparse


def calc_equilibrium_distr(T_sparse):
    """Calculate the equilibrium distribution for the given transition matrix

    Calculates the first eigen value near 1.+0i and the corresponding eigen
    vector which is equal to the equilibrium distribution of the two neuron
    system.
    """
    eig_val, eig_vec = eigs(T_sparse, k=1, sigma=1.0 + 0.0j)
    logging.info(("eigen value:", eig_val))
    # find eigen value = 1.
    equil_id = np.where(np.isclose(eig_val, 1.0 + 0.0j))[0][0]
    equil_distr = np.real(eig_vec[:, equil_id]).flatten()
    all_real = np.all(np.imag(equil_distr) == 0.0)
    logging.info(("All values of the eigen vector are real:", all_real))
    equil_distr /= np.sum(equil_distr)

    return equil_distr


def create_nonspiking_trans_matrix(T_sparse, t_max):
    """
    Returns a copy of the transition matrix where all entries are zero that
    would let either neuron spike.
    """
    T_tilde = T_sparse.copy()

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

    # print(T_tilde.toarray())
    T_tilde = T_tilde.tocsr()
    return T_tilde


def create_trans_matrix_neuron_2(T_sparse, t_max):
    """Returns a version of T where neuron 2 will spike.

    Used to calc the RIGHT side of the dt-distribution (dt > 0)
    The dimention of the matrix is (t_max, t_max**2) --> rho = T * q, with rho
    being the desired dt-distribution.
    """
    T_hat_right = sm.dok_matrix((t_max, t_max**2))
    for ceta_1 in range(t_max):
        # if neuron 2 spikes, copy line into T_hat
        i = ceta_1 * t_max
        T_hat_right[ceta_1, :] = T_sparse[i, :]
    # print(T_hat.toarray())
    T_hat_right = T_hat_right.tocsr()
    return T_hat_right


def create_trans_matrix_neuron_1(T_sparse, t_max):
    """Returns a version of T where neuron 1 will spike.

    Used to calc the LEFT side of the dt-distribution (dt < 0)
    The dimention of the matrix is (t_max, t_max**2) --> rho = T * q, with rho
    being the desired dt-distribution.
    """
    T_hat_left = sm.dok_matrix((t_max, t_max**2))
    for ceta_1 in range(t_max):
        # if neuron 1 spikes, copy line into T_hat
        T_hat_left[ceta_1, :] = T_sparse[ceta_1, :]
    # print(T_hat.toarray())
    T_hat_left = T_hat_left.tocsr()
    return T_hat_left


def calc_half_histogram(distr, T_tilde, T_hat, t_max):
    """Calculate one half of the dt distribution

    Takes a starting distribution, which determines if the left or right side
    of the histogram will be calculated.
    """
    dt_hist = np.zeros(t_max)

    # calc the distribution of next time step when we allow neuron 2 or 1 to
    # spike:
    distr_spike = T_hat.dot(distr)
    dt_hist += distr_spike

    for t in range(t_max):
        # new distribution after one update where spikes are forbidden
        distr = T_tilde.dot(distr)
        # calc the distribution of next time step when we allow neuron 2 to
        # spike:
        distr_spike = T_hat.dot(distr)
        dt_hist += distr_spike

    return dt_hist


def calc_dt_distr(PSP, tau_ref, t_max, w_12, w_21, b_1, b_2, tau_syn):
    """Calculates the complete histogram of the 2 neuron system

    Takes the parameters of the network (tau_ref, t_max, w_12, w_21, b_1, b_2, tau_syn)
    and the PSP shape (function) and calculates the whole distribution of the
    interpike intervalls between two neurons.
    """
    # Create transition matrix:
    logging.info("Create transition matrix...")
    T = create_transition_matrix(PSP, tau_ref, t_max, w_12, w_21, b_1, b_2, tau_syn)

    # Create non-spiking transition matrix:
    logging.info("Create non spiking transition matrix.")
    T_nospike = create_nonspiking_trans_matrix(T, t_max)

    # Create transition matrix for right side:
    logging.info("Create transition matrix for right side.")
    T_right = create_trans_matrix_neuron_2(T, t_max)

    # Create transition matrix for right side:
    logging.info("Create transition matrix for left side.")
    T_left = create_trans_matrix_neuron_1(T, t_max)

    # Calculate equilibrium distribution:
    logging.info("Calc the equilibrium distribution.")
    equil_distr = calc_equilibrium_distr(T)

    # Calc the right side of the histogram:
    logging.info("Calc right side of histogram")
    distr_right = np.copy(equil_distr)
    # set initial distribution to states in which neuron one spikes:
    distr_right[t_max:] = 0.0
    # renormalize distribution:
    distr_right /= np.sum(distr_right)

    hist_right = calc_half_histogram(distr_right, T_nospike, T_right, t_max)

    # Calc the left side of the histogram:
    logging.info("Calc left side of histogram")
    distr_left = np.zeros_like(equil_distr)
    # set initial distribution to states in which neuron two spikes:
    for i in range(t_max):
        distr_left[i * t_max] = equil_distr[i * t_max]
    # renormalize distribution:
    distr_left /= np.sum(distr_left)

    hist_left = calc_half_histogram(distr_left, T_nospike, T_left, t_max)

    dt_left_norm = hist_left / np.sum(hist_left) * 0.5
    dt_right_norm = hist_right / np.sum(hist_right) * 0.5
    dt_hist_norm = np.concatenate((dt_left_norm[:0:-1], dt_right_norm))

    return dt_hist_norm


def main():
    """DOCSTRING."""
    PSP = exp_PSP
    logging.info("No simulation data.")
    tau_bio = 10.0
    tau_ref = 50
    tau_syn = 25
    t_max = 150
    w_12 = 0.3
    w_21 = -0.7
    b_1 = 1.3
    b_2 = 0.2
    logging.info("")
    logging.info(
        "\n".join(
            [
                "tau_ref = {}, t_max = {}".format(tau_ref, t_max),
                "b_1 = {}, b_2 = {}".format(b_1, b_2),
                "w_12 = {}, w_21 = {}".format(w_12, w_21),
            ]
        )
    )

    hist_analytical = calc_dt_distr(PSP, tau_ref, t_max, w_12, w_21, b_1, b_2, tau_syn)

    hist_analytical *= tau_ref / tau_bio
    logging.info(str(hist_analytical))
    xs = np.arange(-t_max + 1, t_max) / tau_ref * tau_bio

    fig, ax = plt.subplots()
    ax.plot(xs, hist_analytical, label="matrix method")
    ax.set_xlim(-30, 30)
    ax.set_xlabel(r"$\Delta t_{pp}$ [ms]")
    ax.set_ylabel(r"$\rho(\Delta t$")
    ax.legend()
    ax.grid()
    ax.set_title(
        "b1={}, b_2={}, w12={}, w21={}, tau_ref={}".format(
            b_1, b_2, w_12, w_21, tau_ref
        )
    )
    file_name = "histogram.png"
    fig.savefig(file_name, dpi=150)
    logging.info("Histogram saved to {}".format(file_name))


if __name__ == "__main__":
    main()
