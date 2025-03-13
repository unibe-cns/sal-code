"""Transfer matrix for the two neuron buesing model

Defines a class with generators for the creation of the transfer matrix T. The
T can cope with every kind of PSP shapes.
"""

import numpy as np


def sigma(u, tau):
    return 1 / (1 + tau * np.exp(-u))


def block_A(zeta_1, PSP, tau_ref, t_max, w_12, w_21, b_1, b_2, tau_syn):
    for zeta_2 in range(1, tau_ref):
        yield (zeta_2, zeta_2 - 1, 1.0)

    res_1 = sigma(b_2 + w_21 * PSP(zeta_1, tau_ref, tau_syn), tau_ref)
    res_2 = 1 - res_1
    for zeta_2 in range(tau_ref, t_max):
        yield (0, zeta_2 - 1, res_1)
        yield (zeta_2, zeta_2 - 1, res_2)

    res = sigma(b_2 + w_21 * PSP(zeta_1, tau_ref, tau_syn), tau_ref)
    yield (0, t_max - 1, res)
    res = 1 - sigma(b_2 + w_21 * PSP(zeta_1, tau_ref, tau_syn), tau_ref)
    yield (t_max - 1, t_max - 1, res)


def block_B(zeta_1, PSP, tau_ref, t_max, w_12, w_21, b_1, b_2, tau_syn):
    for zeta_2 in range(1, tau_ref):
        res = sigma(b_1 + w_12 * PSP(zeta_2, tau_ref, tau_syn), tau_ref)
        yield (zeta_2, zeta_2 - 1, res)

    for zeta_2 in range(tau_ref, t_max):
        res_1 = sigma(b_1 + w_12 * PSP(zeta_2, tau_ref, tau_syn), tau_ref) * sigma(
            b_2 + w_21 * PSP(zeta_1, tau_ref, tau_syn), tau_ref
        )
        res_2 = sigma(b_1 + w_12 * PSP(zeta_2, tau_ref, tau_syn), tau_ref) * (
            1 - sigma(b_2 + w_21 * PSP(zeta_1, tau_ref, tau_syn), tau_ref)
        )
        yield (0, zeta_2 - 1, res_1)
        yield (zeta_2, zeta_2 - 1, res_2)

    yield (0, t_max - 1, res_1)
    yield (t_max - 1, t_max - 1, res_2)


def block_C(zeta_1, PSP, tau_ref, t_max, w_12, w_21, b_1, b_2, tau_syn):
    for zeta_2 in range(1, tau_ref):
        res = 1 - sigma(b_1 + w_12 * PSP(zeta_2, tau_ref, tau_syn), tau_ref)
        yield (zeta_2, zeta_2 - 1, res)

    for zeta_2 in range(tau_ref, t_max):
        res_1 = (
            1 - sigma(b_1 + w_12 * PSP(zeta_2, tau_ref, tau_syn), tau_ref)
        ) * sigma(b_2 + w_21 * PSP(zeta_1, tau_ref, tau_syn), tau_ref)
        yield (0, zeta_2 - 1, res_1)
        res_2 = (1 - sigma(b_1 + w_12 * PSP(zeta_2, tau_ref, tau_syn), tau_ref)) * (
            1 - sigma(b_2 + w_21 * PSP(zeta_1, tau_ref, tau_syn), tau_ref)
        )
        yield (zeta_2, zeta_2 - 1, res_2)

    yield (0, t_max - 1, res_1)
    yield (t_max - 1, t_max - 1, res_2)


def transition_matrix(PSP, tau_ref, t_max, w_12, w_21, b_1, b_2, tau_syn):
    # apply blockmatrix A
    for start_A in range(tau_ref - 1):
        for i, j, A in block_A(
            start_A + 1, PSP, tau_ref, t_max, w_12, w_21, b_1, b_2, tau_syn
        ):
            id_i = (start_A + 1) * t_max + i
            id_j = start_A * t_max + j
            yield (id_i, id_j, A)

    # apply blockmatrix B and C
    for start_B in range(tau_ref - 1, t_max - 1):
        for i, j, B in block_B(
            start_B + 1, PSP, tau_ref, t_max, w_12, w_21, b_1, b_2, tau_syn
        ):
            id_i = 0 + i
            id_j = start_B * t_max + j
            yield (id_i, id_j, B)
        for i, j, C in block_C(
            start_B + 1, PSP, tau_ref, t_max, w_12, w_21, b_1, b_2, tau_syn
        ):
            id_i = (start_B + 1) * t_max + i
            id_j = start_B * t_max + j
            yield (id_i, id_j, C)

    for i, j, B in block_B(
        start_B + 1, PSP, tau_ref, t_max, w_12, w_21, b_1, b_2, tau_syn
    ):
        id_i = 0 + i
        id_j = (t_max - 1) * t_max + j
        yield (id_i, id_j, B)
    for i, j, C in block_C(
        start_B + 1, PSP, tau_ref, t_max, w_12, w_21, b_1, b_2, tau_syn
    ):
        id_i = (t_max - 1) * t_max + i
        id_j = (t_max - 1) * t_max + j
        yield (id_i, id_j, C)
