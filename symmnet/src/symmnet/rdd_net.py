"""Define the architecture of the spiking RDD net."""

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn

from .rdd_layers import SpikingFA


class RDDNetBase:
    """Base class for RDD-based spiking networks of arbitrary depth.

    Builds a chain of SpikingFA layers and implements the shared forward pass,
    reset, and feedback-weight update logic. Subclasses add weight I/O with
    external PyTorch models (see RDDNet).
    """

    def __init__(self, layer_dims: list[int]) -> None:
        self.n_layers = len(layer_dims)
        assert self.n_layers >= 2, "The network needs at least two layers!"
        self.classification_layers = []

        self.classification_layers.append(SpikingFA(layer_dims[0], None, layer_dims[1]))
        for i in range(1, self.n_layers - 1):
            self.classification_layers.append(
                SpikingFA(layer_dims[i], layer_dims[i - 1], layer_dims[i + 1])
            )
        self.classification_layers.append(SpikingFA(layer_dims[-1], layer_dims[-2]))

    def out(self, *args: npt.NDArray | None) -> None:
        """Update all layers for one timestep given external driving inputs.

        Args:
            *args: One driving spike history array per non-output layer, i.e.
                ``len(args) == len(classification_layers) - 1``. Pass ``None``
                for layers that receive no external drive at this timestep.
        """
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

    def reset(self) -> None:
        """
        Resets the state of all SpikingFA layers (membrane potentials, spike history, etc.).
        """
        for layer in self.classification_layers:
            layer.reset()

    def update_fb_weights(self) -> None:
        """
        Updates the feedback weights in all but the last SpikingFA layer.
        """
        for layer in self.classification_layers[:-1]:
            layer.update_fb_weights()


class RDDNet(RDDNetBase):
    """
    Network composed of multiple SpikingFA layers, each optionally using
    Regression Discontinuity Design (RDD) logic for causal inference of feedback
    weights.

    Extends RDDNetBase with methods to copy weights between PyTorch layers and
    the SpikingFA layers.

    Attributes:
        classification_layers (list): List of SpikingFA layers representing the network.
    """

    def __init__(self, layer_dims: list[int]) -> None:
        """Initializes the RDDNet with SpikingFA layers."""
        super().__init__(layer_dims)

    def copy_weights_from(self, layers: list[nn.Module]) -> None:
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

    def copy_weights_to(
        self, layers: list[nn.Module], device: torch.device | str
    ) -> None:
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
