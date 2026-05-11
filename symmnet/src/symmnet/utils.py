#!/usr/bin/env python3


import math

import torch
from torch import Tensor


def batched_outer(a: Tensor, b: Tensor) -> Tensor:
    return a.unsqueeze(2) * b.unsqueeze(1)


def asym_var(a: Tensor, b: Tensor) -> Tensor:
    return (a - b).var()


def asym_angle(a: Tensor, b: Tensor) -> Tensor:
    assert a.shape == b.shape
    aa = a.flatten()
    bb = b.flatten()
    cos = torch.dot(aa, bb) / (aa.norm() * bb.norm())
    return torch.acos(torch.clamp(cos, -1.0, 1.0)) * 180.0 / math.pi


def corrcoef(a: Tensor, b: Tensor) -> Tensor:
    assert a.shape == b.shape
    return torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1]
