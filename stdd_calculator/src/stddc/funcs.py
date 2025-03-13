#!/usr/bin/env python3

from typing import Union

import numpy as np


def rect_PSP(zeta, tau_ref, tau_syn):
    """rectangular psp shape."""
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


def _exp_window(
    dt: Union[float, int],
    a_plus: float,
    a_minus: float,
    tau_plus: float,
    tau_minus: float,
) -> float:
    if dt > 0.0:
        return a_plus * np.exp(-dt / tau_plus)
    elif dt < 0.0:
        return a_minus * np.exp(dt / tau_minus)
    else:
        return 0.0


vrect_PSP = np.vectorize(rect_PSP)
vexp_PSP = np.vectorize(exp_PSP)
vcutoff_PSP = np.vectorize(cutoff_PSP)
vtail_PSP = np.vectorize(tail_PSP)
valpha_PSP = np.vectorize(alpha_PSP)

exp_window = np.vectorize(_exp_window)
