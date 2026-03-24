#!/usr/bin/env python3

import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .utils import batched_outer

# TODO: add a non-rolling buffer and an all-at-once stdp rule
# TODO: maybe register all relevant model params (time constants etc.) as
# buffers with self.register_buffer?? --> makes it easier to log an entire model
# for reporducibility


class BaseSpikeBuffer(ABC, nn.Module):
    """
    Base class for a rolling binary spike buffer.
    """

    def __init__(
        self,
        num_neurons: int,
        buffer_length: int,
        batch_size: int = 1,
    ):
        super().__init__()
        self.buffer_length = buffer_length
        self.register_buffer(
            "buffer", torch.zeros(batch_size, num_neurons, buffer_length)
        )

    @abstractmethod
    def append(self, spikes):
        pass

    @abstractmethod
    def get(self):
        pass


class SimpleSpikeBuffer(BaseSpikeBuffer):
    """The simplest buffer one could think of."""

    def append(self, spikes):
        self.buffer = torch.roll(self.buffer, shifts=1, dims=2)
        self.buffer[:, :, 0] = spikes

    def get(self):
        return self.buffer


class BaseKernel(nn.Module):
    """Base kernel, both for PSP and STDP."""

    def __init__(self, kernel):
        super().__init__()
        assert kernel.shape[0] == 1, "Dimension 0 of the Kernel has to be 1!"
        assert kernel.shape[1] == 1, "Dimension 1 of the Kernel has to be 1!"
        assert len(kernel.shape) == 3, "The kernel must have 3 dimensions!"
        self.register_buffer("kernel", kernel)
        self.len_kernel = self.kernel.shape[2]

    def forward(self, spikes):
        # spikes: [batch_size, num_neurons, time_steps]
        # Use 1D convolution to apply the box filter along the time axis
        # Reshape for conv1d: [batch_size * num_neurons, 1, time_steps]
        batch_size, num_neurons, time_steps = spikes.shape
        x = spikes.reshape(batch_size * num_neurons, 1, time_steps)
        # Apply convolution with padding='same' emulation
        conv = torch.nn.functional.conv1d(x, self.kernel, padding=self.len_kernel - 1)
        # Reshape back to [batch_size, num_neurons, time_steps]
        # important: pytorch adds some padding, so throw away the "future"-side
        return conv.reshape(batch_size, num_neurons, -1)[:, :, self.len_kernel - 1 :]


class RectangularPSP(BaseKernel):
    """Rectangular PSPs."""

    def __init__(self, t_syn):
        kernel = torch.ones(1, 1, t_syn)
        super().__init__(kernel)


class AlphaPSP(BaseKernel):
    """Alpha-shaped PSP kernel, normalized to t_ref."""

    def __init__(self, t_syn, t_ref):
        ts = torch.arange(int(t_syn * 10)).reshape(1, 1, -1)
        kernel = t_ref / t_syn**2 * ts * torch.exp(-ts / t_syn)
        super().__init__(kernel)


class ExpSTDP(BaseKernel):
    """Exponential STDP window."""

    def __init__(self, tau, a, len_kernel):
        ts = torch.arange(len_kernel).reshape(1, 1, -1)
        kernel = a * torch.exp(-ts / tau)
        super().__init__(kernel)


class ActivationFunction(nn.Module):
    def __init__(self, t_ref):
        super().__init__()
        self.log_t_ref = math.log(t_ref)

    def forward(self, mem_pot):
        return torch.sigmoid(mem_pot - self.log_t_ref)


class GLMHiddenLayer(nn.Module):
    def __init__(
        self,
        n_in,
        n_out,
        n_next,
        t_ref,
        stdp_lr,
        batch_size=1,
        buffer_length=2,
        psp=None,
    ):
        super().__init__()

        # model settings:
        self.n_in = n_in
        self.n_out = n_out
        self.n_next = n_next
        self.t_ref = t_ref
        self.stdp_lr = stdp_lr

        # define parameters:
        self.weight = nn.Parameter(torch.empty(n_out, n_in))
        self.fb_weight = nn.Parameter(torch.empty(n_out, n_next))
        self.bias = nn.Parameter(torch.zeros(n_out))
        self.init_params()

        # register submodules
        self.psp = RectangularPSP(t_ref) if psp is None else psp
        self.phi = ActivationFunction(t_ref)
        self.causal_stdp = ExpSTDP(t_ref, -1.0, buffer_length * t_ref)
        self.anticausal_stdp = ExpSTDP(t_ref, 1.0, buffer_length * t_ref)

        # internal states:
        self.register_buffer("mem_pot", torch.zeros(batch_size, n_out))
        self.spikes = SimpleSpikeBuffer(
            n_out, buffer_length * t_ref, batch_size=batch_size
        )  # TODO: change buffer length!
        self.register_buffer(
            "last_spike_counter", torch.full_like(self.mem_pot, torch.inf)
        )
        self.register_buffer("delta_fb_weight", torch.zeros(batch_size, n_out, n_next))

        # others:
        self.batch_size = batch_size

    @property
    def device(self):
        return next(self.parameters()).device

    def init_params(self):
        # initialize params (although they're actually copied from the corresponding ANN)
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.fb_weight, a=5**0.5)
        bound = self.n_out**-0.5  # TODO check if this is useful
        nn.init.uniform_(self.bias, -bound, bound)

    def update_mempot(self, bottom_up_spikes, top_down_spikes):
        bottom_up_psps = self.psp(bottom_up_spikes)
        top_down_psps = self.psp(top_down_spikes)
        self.mem_pot = (
            torch.matmul(bottom_up_psps[:, :, 0], self.weight.t())
            + torch.matmul(top_down_psps[:, :, 0], self.fb_weight.t())
            + self.bias
        )

    def update_spikes(self, t=0.0, start_id=1):
        inst_rate = self.phi(self.mem_pot)
        random_vals = torch.rand(self.batch_size, self.n_out, device=self.device)
        new_spikes = torch.logical_and(
            inst_rate > random_vals, self.last_spike_counter > self.t_ref
        )
        self.last_spike_counter += 1.0
        self.last_spike_counter[new_spikes] = 1.0
        self.spikes.append(new_spikes.float())

    def forward(self):
        # not sure, what's the best implementation here...
        pass

    def fb_stdp_online(self, top_down_spikes):
        # here: top_down_spikes = pre, own spikes = post
        trace_pre = self.causal_stdp(top_down_spikes)
        trace_post = self.anticausal_stdp(self.spikes.get())
        # computes the outer product and adds it to delta_fb_weight
        self.delta_fb_weight += batched_outer(
            self.spikes.get()[:, :, 0], trace_pre[:, :, 0]
        )
        self.delta_fb_weight += batched_outer(
            trace_post[:, :, 0], top_down_spikes[:, :, 0]
        )

    def apply_fb_weight_update(self, zero_dw=True):
        dw = self.delta_fb_weight.mean(dim=0)
        dw /= float(self.causal_stdp.len_kernel)
        with torch.no_grad():
            self.fb_weight += self.stdp_lr * dw
            if zero_dw:
                self.delta_fb_weight.zero_()
        return dw  # NOTE for debugging purposes


class GLMInputLayer(GLMHiddenLayer):
    def __init__(
        self,
        n_out,
        n_next,
        t_ref,
        stdp_lr,
        batch_size=1,
        buffer_length=2,
        psp=None,
    ):
        super().__init__(
            0,
            n_out,
            n_next,
            t_ref,
            stdp_lr,
            batch_size=batch_size,
            buffer_length=buffer_length,
            psp=psp,
        )
        self.weight = None

    def init_params(self):
        # initialize params (although they're actually copied from the corresponding ANN)
        nn.init.kaiming_uniform_(self.fb_weight, a=5**0.5)
        bound = self.n_out**-0.5  # TODO check if this is useful
        nn.init.uniform_(self.bias, -bound, bound)

    def update_mempot(self, top_down_spikes):
        top_down_psps = self.psp(top_down_spikes)
        self.mem_pot = (
            torch.matmul(top_down_psps[:, :, 0], self.fb_weight.t()) + self.bias
        )


class GLMOutputLayer(GLMHiddenLayer):
    def __init__(
        self,
        n_in,
        n_out,
        t_ref,
        batch_size=1,
        buffer_length=2,
        psp=None,
    ):
        super().__init__(
            n_in,
            n_out,
            0,
            t_ref,
            0.0,
            batch_size=batch_size,
            buffer_length=buffer_length,
            psp=psp,
        )
        self.fb_weight = None
        self.delta_fb_weight = None
        self.stdp_lr = None

    def init_params(self):
        # initialize params (although they're actually copied from the corresponding ANN)
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        bound = self.n_out**-0.5  # TODO check if this is useful
        nn.init.uniform_(self.bias, -bound, bound)

    def update_mempot(self, bottom_up_spikes):
        bottom_up_psps = self.psp(bottom_up_spikes)
        self.mem_pot = (
            torch.matmul(bottom_up_psps[:, :, 0], self.weight.t()) + self.bias
        )

    def fb_stdp_online(self, bottom_up_spikes):
        raise NotImplementedError(
            "GLMOutputLayer doesn't have feedback weights and hence no stdp rule for it."
        )

    def apply_fb_weight_update(self):
        raise NotImplementedError(
            "GLMOutputLayer doesn't have feedback weights and hence no stdp rule for it."
        )


class SALNetBase(nn.Module):
    def __init__(
        self,
        layer_dims,
        t_ref,
        stdp_lr,
        batch_size=1,
        buffer_length=2,
        psp=None,
    ):
        super().__init__()
        self.n_layers = len(layer_dims)
        assert self.n_layers >= 2, "The network needs at least two layers!"

        if isinstance(stdp_lr, float):
            self.stdp_lr = [stdp_lr] * (self.n_layers - 1)
        elif isinstance(stdp_lr, list) and len(stdp_lr) == self.n_layers - 1:
            self.stdp_lr = stdp_lr
        else:
            raise TypeError

        layers = [
            GLMInputLayer(
                n_out=layer_dims[0],
                n_next=layer_dims[1],
                t_ref=t_ref,
                stdp_lr=self.stdp_lr[0],
                batch_size=batch_size,
                buffer_length=buffer_length,
                psp=psp,
            ),
        ]
        for i in range(1, self.n_layers - 1):
            layers.append(
                GLMHiddenLayer(
                    n_in=layer_dims[i - 1],
                    n_out=layer_dims[i],
                    n_next=layer_dims[i + 1],
                    t_ref=t_ref,
                    stdp_lr=self.stdp_lr[i],
                    batch_size=batch_size,
                    buffer_length=buffer_length,
                    psp=psp,
                ),
            )

        layers.append(
            GLMOutputLayer(
                n_in=layer_dims[self.n_layers - 2],
                n_out=layer_dims[self.n_layers - 1],
                t_ref=t_ref,
                batch_size=batch_size,
                buffer_length=buffer_length,
                psp=psp,
            ),
        )
        self.layers = nn.ModuleList(layers)

    def update_mempot(self):
        # get spikes
        spikes = [layer.spikes.get() for layer in self.layers]
        # update layers
        self.layers[0].update_mempot(spikes[1])
        for i in range(1, self.n_layers - 1):
            self.layers[i].update_mempot(spikes[i - 1], spikes[i + 1])
        self.layers[-1].update_mempot(spikes[-2])

    def update_spikes(self, t=0.0):
        for layer in self.layers:
            layer.update_spikes()

    def fb_stdp_online(self):
        for i in range(self.n_layers - 1):
            self.layers[i].fb_stdp_online(self.layers[i + 1].spikes.get())

    def apply_fb_weight_update(self):
        dws = []  # TODO: for debugging
        for layer in self.layers[:-1]:
            dw = layer.apply_fb_weight_update()
            dws.append(dw)
        return dws

    def set_stdp_lr(self, stdp_lr):
        if isinstance(stdp_lr, float):
            self.stdp_lr = [stdp_lr] * 3
        for layer, lr in zip(self.layers[:-1], self.stdp_lr):
            layer.stdp_lr = lr


class SALNet(SALNetBase):
    _COMMON_LAYERS = [
        ("fc1.weight", "layers.1.weight"),
        ("fc1.fb_weight", "layers.0.fb_weight"),
        ("fc1.bias", "layers.1.bias"),
        ("fc2.weight", "layers.2.weight"),
        ("fc2.fb_weight", "layers.1.fb_weight"),
        ("fc2.bias", "layers.2.bias"),
        ("fc3.weight", "layers.3.weight"),
        ("fc3.fb_weight", "layers.2.fb_weight"),
        ("fc3.bias", "layers.3.bias"),
    ]

    def __init__(
        self,
        layer_dims,
        t_ref,
        stdp_lr,
        batch_size=1,
        buffer_length=2,
        psp=None,
    ):
        assert (
            len(layer_dims) == 4
        ), "SALNT needs 4 layers to be compatible with the corresponding ConvNet."
        super().__init__(
            layer_dims,
            t_ref,
            stdp_lr,
            batch_size=batch_size,
            buffer_length=buffer_length,
            psp=psp,
        )

    def load_common_state_dict(self, conv_net_state_dict):
        own_state_dict = self.state_dict()
        for other, own in SALNet._COMMON_LAYERS:
            # transpose the feedback weights, because I erroneously definedy the
            # sal fb_weights the other way around...
            if "fb_weight" in other:
                own_state_dict[own] = conv_net_state_dict[other].t()
            else:
                own_state_dict[own] = conv_net_state_dict[other]
        self.load_state_dict(own_state_dict, strict=False)

    def get_common_state_dict(self):
        other_state_dict = {}
        own_state_dict = self.state_dict()
        for other, own in SALNet._COMMON_LAYERS:
            if "fb_weight" in other:
                other_state_dict[other] = own_state_dict[own].t()
            else:
                other_state_dict[other] = own_state_dict[own]
        return other_state_dict
