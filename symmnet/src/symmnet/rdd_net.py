"""Define the architecture of the spiking RDD net."""

import numpy as np
import torch

from .rdd_layers import SpikingFA


class RDDNetBase:
    """This is new!"""

    def __init__(self, layer_dims):
        self.n_layers = len(layer_dims)
        assert self.n_layers >= 2, "The network needs at least two layers!"
        self.classification_layers = []

        self.classification_layers.append(SpikingFA(layer_dims[0], None, layer_dims[1]))
        for i in range(1, self.n_layers - 1):
            self.classification_layers.append(
                SpikingFA(layer_dims[i], layer_dims[i - 1], layer_dims[i + 1])
            )
        self.classification_layers.append(SpikingFA(layer_dims[-1], layer_dims[-2]))

    def out(self, *args):
        """args must be the driving_spike_hist"""
        assert len(args) == len(self.classification_layers) - 1

        # Layer 0: receives external drive and feedback from layer 1
        self.classification_layers[0].update(
            None,
            self.classification_layers[1].spike_hist,
            driving_input=args[0],
        )

        # intermediate layers: receives feedforward from previous layer, feedback from layer next, and external drive
        for i in range(1, self.n_layers - 1):
            self.classification_layers[i].update(
                self.classification_layers[i - 1].spike_hist,
                self.classification_layers[i + 1].spike_hist,
                driving_input=args[i],
            )

        # last Layer: receives feedforward from penultimate layer, no feedback, no external drive
        self.classification_layers[-1].update(self.classification_layers[-2].spike_hist)

    def reset(self):
        """
        Resets the state of all SpikingFA layers (membrane potentials, spike history, etc.).
        """
        for layer in self.classification_layers:
            layer.reset()

    def update_fb_weights(self):
        """
        Updates the feedback weights in all but the last SpikingFA layer.
        """
        for layer in self.classification_layers[:-1]:
            layer.update_fb_weights()


class RDDNet:
    """
    Network composed of multiple SpikingFA layers, each optionally using
    Regression Discontinuity Design (RDD) logic for causal inference of feedback
    weigts.

    This class provides methods to copy weights between PyTorch layers and
    the SpikingFA layers, perform sequential updates, and manage feedback weights.

    Attributes:
        classification_layers (list): List of SpikingFA layers representing the network.

    TODO: make this inherit from RDDNetBase!
    """

    def __init__(self, layer_dims):
        """
        Initializes the RDDNet with a fixed architecture of SpikingFA layers.
        """
        self.classification_layers = []

        self.classification_layers.append(SpikingFA(layer_dims[0], None, layer_dims[1]))
        self.classification_layers.append(
            SpikingFA(layer_dims[1], layer_dims[0], layer_dims[2])
        )
        self.classification_layers.append(
            SpikingFA(layer_dims[2], layer_dims[1], layer_dims[3])
        )
        self.classification_layers.append(SpikingFA(layer_dims[3], layer_dims[2]))

    def copy_weights_from(self, layers):
        """
        Copies weights and feedback weights from the linear layers of the
        corresponting pytorch ANN.

        Args:
            layers (list): List of PyTorch layers (e.g., LinearFA/Conv2dFA).
        """
        # Copy only feedback weights for the first SpikingFA layer
        self.classification_layers[0].set_weights(
            None, None, layers[2].fb_weight.detach().cpu().numpy().astype(np.float32).T
        )
        # Copy weights, biases, and feedback weights for the next layers
        self.classification_layers[1].set_weights(
            layers[2].weight.detach().cpu().numpy().astype(np.float32),
            layers[2].bias.detach().cpu().numpy().astype(np.float32)[:, np.newaxis],
            layers[3].fb_weight.detach().cpu().numpy().astype(np.float32).T,
        )
        self.classification_layers[2].set_weights(
            layers[3].weight.detach().cpu().numpy().astype(np.float32),
            layers[3].bias.detach().cpu().numpy().astype(np.float32)[:, np.newaxis],
            layers[4].fb_weight.detach().cpu().numpy().astype(np.float32).T,
        )
        self.classification_layers[3].set_weights(
            layers[4].weight.detach().cpu().numpy().astype(np.float32),
            layers[4].bias.detach().cpu().numpy().astype(np.float32)[:, np.newaxis],
        )

    def copy_weights_to(self, layers, device):
        """
        Copies feedback weights from the SpikingFA layers back to the corresponding
        PyTorch layers, typically after RDD-based updates.

        Args:
            layers (list): List of PyTorch layers to receive the feedback weights.
            device (torch.device): Device to move the weights to (CPU/GPU).
        """
        # Only feedback weights are copied back to PyTorch layers
        layers[2].fb_weight.data = torch.from_numpy(
            self.classification_layers[0].fb_weight.astype(np.float32).T
        ).to(device)
        layers[3].fb_weight.data = torch.from_numpy(
            self.classification_layers[1].fb_weight.astype(np.float32).T
        ).to(device)
        layers[4].fb_weight.data = torch.from_numpy(
            self.classification_layers[2].fb_weight.astype(np.float32).T
        ).to(device)

    def out(self, driving_spike_hist_1, driving_spike_hist_2, driving_spike_hist_3):
        """
        Sequentially updates each SpikingFA layer given the driving spike histories
        and the spike histories of adjacent layers.

        Args:
            driving_spike_hist_1 (np.ndarray): External driving input for the first layer.
            driving_spike_hist_2 (np.ndarray): External driving input for the second layer.
            driving_spike_hist_3 (np.ndarray): External driving input for the third layer.
        """
        # Layer 0: receives external drive and feedback from layer 1
        self.classification_layers[0].update(
            None,
            self.classification_layers[1].spike_hist,
            driving_input=driving_spike_hist_1,
        )
        # Layer 1: receives feedforward from layer 0, feedback from layer 2, and external drive
        self.classification_layers[1].update(
            self.classification_layers[0].spike_hist,
            self.classification_layers[2].spike_hist,
            driving_input=driving_spike_hist_2,
        )
        # Layer 2: receives feedforward from layer 1, feedback from layer 3, and external drive
        self.classification_layers[2].update(
            self.classification_layers[1].spike_hist,
            self.classification_layers[3].spike_hist,
            driving_input=driving_spike_hist_3,
        )
        # Layer 3: receives feedforward from layer 2, no feedback, no external drive
        self.classification_layers[3].update(self.classification_layers[2].spike_hist)

    def reset(self):
        """
        Resets the state of all SpikingFA layers (membrane potentials, spike history, etc.).
        """
        for layer in self.classification_layers:
            layer.reset()

    def update_fb_weights(self):
        """
        Updates the feedback weights in all but the last SpikingFA layer.
        """
        for layer in self.classification_layers[:-1]:
            layer.update_fb_weights()
