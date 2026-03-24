"""Conventional ANN layers based on pytorch."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter


def conv2d_fa_backward_hook(module, grad_input, grad_output):
    """
    Backward hook for Conv2dFA layers.

    Computes the gradient for the input using fixed random feedback weights
    instead of the transposed forward weights.

    Args:
        module (nn.Module): The Conv2dFA module.
        grad_input (tuple): Gradients with respect to the input of the layer.
        grad_output (tuple): Gradients with respect to the output of the layer.

    Returns:
        tuple: Modified gradients with feedback alignment applied.
    """
    if grad_input[0] is not None:
        grad_input_fa = torch.nn.grad.conv2d_input(
            grad_input[0].size(),
            module.fb_weight,
            grad_output[0],
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
        )

        return (grad_input_fa,) + grad_input[1:]


class LinearFuncFA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, weight, bias, fb_weight):
        ctx.save_for_backward(inp, weight, bias, fb_weight)
        return F.linear(inp, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        inp, weight, bias, fb_weight = ctx.saved_tensors

        grad_input = grad_output.mm(fb_weight)
        # summing over batches happens implicitely in mm:
        grad_weight = grad_output.t().mm(inp)
        # sum over batches:
        grad_bias = grad_output.sum(0) if bias is not None else None
        # B is fixed, does not recieve gradients
        return grad_input, grad_weight, grad_bias, None


class LinearFuncKP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, weight, bias, fb_weight):
        ctx.save_for_backward(inp, weight, bias, fb_weight)
        return F.linear(inp, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        inp, weight, bias, fb_weight = ctx.saved_tensors

        grad_input = grad_output.mm(fb_weight)
        # summing over batches happens implicitely in mm:
        grad_weight = grad_output.t().mm(inp)
        # sum over batches:
        grad_bias = grad_output.sum(0) if bias is not None else None
        # grad B is the same as grad W
        grad_fb_weight = grad_output.t().mm(inp)
        return grad_input, grad_weight, grad_bias, grad_fb_weight


class LinearFuncSCFA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, weight, bias, fb_weight):
        ctx.save_for_backward(inp, weight, bias, fb_weight)
        return F.linear(inp, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        inp, weight, bias, fb_weight = ctx.saved_tensors

        grad_input = grad_output.mm(fb_weight * weight.sign())
        # summing over batches happens implicitely in mm:
        grad_weight = grad_output.t().mm(inp)
        # sum over batches:
        grad_bias = grad_output.sum(0) if bias is not None else None
        # B is fixed, does not recieve gradients
        return grad_input, grad_weight, grad_bias, None


def linear_fa(inp, weight, bias, fb_weights):
    return LinearFuncFA.apply(inp, weight, bias, fb_weights)


def linear_kp(inp, weight, bias, fb_weights):
    return LinearFuncKP.apply(inp, weight, bias, fb_weights)


def linear_scfa(inp, weight, bias, fb_weights):
    return LinearFuncSCFA.apply(inp, weight, bias, fb_weights)


class LinearFA(nn.Module):
    """
    Linear layer with feedback alignment.

    This module implements a linear transformation with random feedback weights
    used in the backward pass, as described in Lillicrap et al., 2016:
    https://www.nature.com/articles/ncomms13276

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        bias (bool, optional): If set to False, the layer will not learn an additive bias. Default: True.
    """

    __constants__ = ["bias", "in_features", "out_features"]

    def __init__(self, in_features, out_features, bias=True):
        """
        Initialize the LinearFA layer.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            bias (bool): If True, adds a learnable bias to the output.
        """
        super(LinearFA, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.fb_weight = Parameter(
            torch.Tensor(out_features, in_features), requires_grad=False
        )
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize or reset the parameters of the layer.
        """
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        init.kaiming_uniform_(self.fb_weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        """
        Forward pass of the LinearFA layer.

        Args:
            input (Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            Tensor: Output tensor of shape (batch_size, out_features).
        """
        return linear_fa(input, self.weight, self.bias, self.fb_weight)

    def extra_repr(self):
        """
        Extra representation of the module for printing.

        Returns:
            str: String representation of the module.
        """
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


class LinearKP(LinearFA):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        self.fb_weight.requires_grad_()

    def forward(self, input):
        """
        Forward pass of the LinearFA layer.

        Args:
            input (Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            Tensor: Output tensor of shape (batch_size, out_features).
        """
        return linear_kp(input, self.weight, self.bias, self.fb_weight)


class LinearSCFA(LinearFA):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        self.fb_weight.data = self.fb_weight.data.abs()

    def forward(self, input):
        """
        Forward pass of the LinearFA layer.

        Args:
            input (Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            Tensor: Output tensor of shape (batch_size, out_features).
        """
        return linear_scfa(input, self.weight, self.bias, self.fb_weight)


class _ConvNdFA(nn.Module):
    """
    N-dimensional convolution layer with feedback alignment.

    This class is based on PyTorch's _ConvNd with the addition of random feedback weights
    for use in the backward pass, as described in Lillicrap et al., 2016:
    https://www.nature.com/articles/ncomms13276

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple): Stride of the convolution.
        padding (int or tuple): Zero-padding added to both sides of the input.
        dilation (int or tuple): Spacing between kernel elements.
        transposed (bool): If True, uses transposed convolution.
        output_padding (int or tuple): Additional size added to one side of the output shape.
        groups (int): Number of blocked connections from input channels to output channels.
        bias (bool): If True, adds a learnable bias to the output.
        padding_mode (str): Type of padding. Default: 'zeros'.
    """

    __constants__ = ["stride", "padding", "dilation", "groups", "bias", "padding_mode"]

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        bias,
        padding_mode,
    ):
        """
        Initialize the _ConvNdFA layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (tuple): Size of the convolving kernel.
            stride (tuple): Stride of the convolution.
            padding (tuple): Padding added to all four sides of the input.
            dilation (tuple): Spacing between kernel elements.
            transposed (bool): If True, uses transposed convolution.
            output_padding (tuple): Additional size added to one side of the output shape.
            groups (int): Number of blocked connections from input channels to output channels.
            bias (bool): If True, adds a learnable bias to the output.
            padding_mode (str): Type of padding. Default: 'zeros'.
        """
        super(_ConvNdFA, self).__init__()
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode

        if transposed:
            # Weight and feedback weight shapes for transposed convolution
            self.weight = Parameter(
                torch.Tensor(in_channels, out_channels // groups, *kernel_size)
            )

            self.fb_weight = Parameter(
                torch.Tensor(in_channels, out_channels // groups, *kernel_size),
                requires_grad=False,
            )
        else:
            # Weight and feedback weight shapes for standard convolution
            self.weight = Parameter(
                torch.Tensor(out_channels, in_channels // groups, *kernel_size)
            )

            self.fb_weight = Parameter(
                torch.Tensor(out_channels, in_channels // groups, *kernel_size),
                requires_grad=False,
            )
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize or reset the parameters of the layer.
        """
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        init.kaiming_uniform_(self.fb_weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        """
        Initialize or reset the parameters of the layer.
        """
        s = (
            "{in_channels}, {out_channels}, kernel_size={kernel_size}"
            ", stride={stride}"
        )
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.output_padding != (0,) * len(self.output_padding):
            s += ", output_padding={output_padding}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is None:
            s += ", bias=False"
        return s.format(**self.__dict__)


class Conv2dFA(_ConvNdFA):
    """
    2D convolution layer with feedback alignment.

    Implements a 2D convolution with random feedback weights in the backward pass,
    as described in Lillicrap et al., 2016:
    https://www.nature.com/articles/ncomms13276

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple, optional): Stride of the convolution. Default: 1.
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0.
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1.
        bias (bool, optional): If True, adds a learnable bias to the output. Default: True.
        padding_mode (str, optional): Type of padding. Default: 'zeros'.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
    ):
        """
        Initialize the Conv2dFA layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int or tuple): Size of the convolving kernel.
            stride (int or tuple): Stride of the convolution.
            padding (int or tuple): Padding added to all four sides of the input.
            dilation (int or tuple): Spacing between kernel elements.
            groups (int): Number of blocked connections from input channels to output channels.
            bias (bool): If True, adds a learnable bias to the output.
            padding_mode (str): Type of padding. Default: 'zeros'.
        """
        # Convert all convolution parameters to 2-element tuples if needed
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2dFA, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            False,
            _pair(0),
            groups,
            bias,
            padding_mode,
        )

        self.register_backward_hook(conv2d_fa_backward_hook)

    def forward(self, input):
        """
        Forward pass of the Conv2dFA layer.

        Args:
            input (Tensor): Input tensor of shape (N, C_in, H_in, W_in).

        Returns:
            Tensor: Output tensor of shape (N, C_out, H_out, W_out).
        """
        if self.padding_mode == "circular":
            # If circular padding is requested, pad manually before convolution
            expanded_padding = (
                (self.padding[1] + 1) // 2,
                self.padding[1] // 2,
                (self.padding[0] + 1) // 2,
                self.padding[0] // 2,
            )
            return F.conv2d(
                F.pad(input, expanded_padding, mode="circular"),
                self.weight,
                self.bias,
                self.stride,
                _pair(0),
                self.dilation,
                self.groups,
            )
        # Standard convolution with specified padding
        return F.conv2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
