#!/usr/bin/env python3

import numpy as np


def rect_PSP(zeta: int, tau_ref: int, tau_syn: int) -> float:
    """Rectangular PSP shape (1 within tau_ref, 0 elsewhere)."""
    return np.heaviside(-zeta + tau_ref, 0.0) * np.heaviside(zeta, 1.0)


def exp_PSP(zeta: int, tau_ref: int, tau_syn: int) -> float:
    """Exponential PSP shape, normalized over [0, inf)."""
    a = (tau_ref / tau_syn) / (1 - np.exp(-tau_ref / tau_syn))
    return a * np.exp(-zeta / tau_syn)


def cutoff_PSP(zeta: int, tau_ref: int, tau_syn: int) -> float:
    """Exponential PSP shape, truncated to zero beyond tau_ref."""
    if zeta <= tau_ref:
        a = (tau_ref / tau_syn) / (1 - np.exp(-tau_ref / tau_syn))
        return a * np.exp(-zeta / tau_syn)
    else:
        return 0.0


def tail_PSP(zeta: int, tau_ref: int, tau_syn: int) -> float:
    """Rectangular PSP within tau_ref, exponential tail beyond."""
    if zeta <= tau_ref:
        return 1.0
    else:
        a = (tau_ref / tau_syn) / (1 - np.exp(-tau_ref / tau_syn))
        return a * np.exp(-zeta / tau_syn)


def alpha_PSP(zeta: int, tau_ref: int, tau_syn: int) -> float:
    """Alpha-function PSP shape (zero at zeta=0, peak near tau_syn)."""
    if zeta > 0.0:
        return tau_ref / tau_syn**2 * zeta * np.exp(-zeta / tau_syn)
    else:
        return 0.0


def _exp_window(
    dt: float | int,
    a_plus: float,
    a_minus: float,
    tau_plus: float,
    tau_minus: float,
) -> float:
    """Asymmetric exponential STDP learning window.

    Args:
        dt: Spike time difference (post minus pre).
        a_plus: Amplitude for dt > 0.
        a_minus: Amplitude for dt < 0.
        tau_plus: Time constant dt > 0.
        tau_minus: Time constant dt < 0.

    Returns:
        Kernel value at dt.
    """
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
