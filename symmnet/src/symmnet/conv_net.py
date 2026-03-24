"""Define the ConvNet architecture (standard pytorch ANN)."""

import math

import torch.nn as nn

from .layers import Conv2dFA, LinearFA, LinearKP, LinearSCFA


def conv2d_outsize(insize, kernel_size):
    return insize - kernel_size + 1


def maxpool2d_outsize(insize, kernel_size, stride):
    return math.floor((insize - kernel_size) / 2 + 1)


def feature_layer_outsize(insize):
    return maxpool2d_outsize(
        conv2d_outsize(maxpool2d_outsize(conv2d_outsize(insize, 5), 2, 2), 5), 2, 2
    )


class ConvNet(nn.Module):
    """
    Convolutional Neural Network with optional feedback alignment layers.

    This network uses Conv2dFA and LinearFA layers when use_backprop is False,
    otherwise uses standard nn.Conv2d and nn.Linear layers.

    Args:
        input_channels (int): Number of input channels.
        use_backprop (bool): Flag to use standard backpropagation layers.
    """

    FC_WEIGHT = ["fc1.weight", "fc2.weight", "fc3.weight"]
    FC_FB_WEIGHT = ["fc1.fb_weight", "fc2.fb_weight", "fc3.fb_weight"]

    def __init__(
        self,
        input_channels,
        input_shape,
        use_backprop=False,
        use_kp=False,
        use_scfa=False,
        use_fa_conv_layers=False,
    ):
        """
        Initialize the ConvNet model.

        Args:
            input_channels (int): Number of input channels.
            use_backprop (bool): If True, use standard backprop layers.
        """
        super(ConvNet, self).__init__()
        n_channels = 64

        feature_layer_outshape = tuple(map(feature_layer_outsize, input_shape))
        fc1_insize = n_channels * math.prod(feature_layer_outshape)
        self.feature_layers_sizes = [fc1_insize, 384, 192, 10]
        self.feature_layers = []

        # Dynamically select linear and convolution layers depending on mode
        if use_backprop:
            Linear = nn.Linear
        elif use_kp and use_scfa:
            raise RuntimeError("'use_kp' and 'use_scfa' are mutually exclusive!")
        elif use_kp and not use_scfa:
            Linear = LinearKP
        elif use_scfa and not use_kp:
            Linear = LinearSCFA
        else:
            Linear = LinearFA
        # optionally use standard fa in conv layers if bp in fc layers to get an
        # "upper bound" for summetrization methods that only work in fc layers.
        Conv2d = nn.Conv2d if use_backprop and not use_fa_conv_layers else Conv2dFA

        # First convolutional block
        self.conv1 = Conv2d(input_channels, n_channels, bias=False, kernel_size=5)
        self.feature_layers.append(self.conv1)
        self.feature_layers.append(nn.ReLU(inplace=True))
        self.feature_layers.append(nn.BatchNorm2d(n_channels))
        self.feature_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # Second convolutional block
        self.conv2 = Conv2d(n_channels, n_channels, bias=False, kernel_size=5)
        self.feature_layers.append(self.conv2)
        self.feature_layers.append(nn.ReLU(inplace=True))
        self.feature_layers.append(nn.BatchNorm2d(n_channels))
        self.feature_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # Flatten features for fully connected layers
        # TODO: I changed this here! Does it still work??
        self.feature_layers.append(nn.Flatten(start_dim=1))

        self.classification_layers = []

        # First fully connected block
        self.fc1 = Linear(self.feature_layers_sizes[0], self.feature_layers_sizes[1])
        self.classification_layers.append(self.fc1)
        self.classification_layers.append(nn.ReLU(inplace=True))
        self.classification_layers.append(nn.BatchNorm1d(self.feature_layers_sizes[1]))

        # Second fully connected block
        self.fc2 = Linear(self.feature_layers_sizes[1], self.feature_layers_sizes[2])
        self.classification_layers.append(self.fc2)
        self.classification_layers.append(nn.ReLU(inplace=True))
        self.classification_layers.append(nn.BatchNorm1d(self.feature_layers_sizes[2]))

        # Output layer
        self.fc3 = Linear(self.feature_layers_sizes[2], self.feature_layers_sizes[3])
        self.classification_layers.append(self.fc3)
        self.classification_layers.append(nn.Softmax(dim=-1))

        # Combine all layers into a single sequential module for the forward pass
        self.out = nn.Sequential(*(self.feature_layers + self.classification_layers))

    def forward(self, x):
        """
        Forward pass of the ConvNet.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_channels, height, width).

        Returns:
            Tensor: Output tensor of shape (batch_size, 10) representing class probabilities.
        """
        return self.out(x)

    def parameters_weight(self):
        for name, param in self.named_parameters():
            if name in ConvNet.FC_WEIGHT:
                yield param

    def parameters_fb_weight(self, ignore_require_grad=False):
        for name, param in self.named_parameters():
            if name in ConvNet.FC_FB_WEIGHT and (
                param.requires_grad or ignore_require_grad
            ):
                yield param

    def parameters_other(self):
        for name, param in self.named_parameters():
            if name not in ConvNet.FC_WEIGHT + ConvNet.FC_FB_WEIGHT:
                yield param
