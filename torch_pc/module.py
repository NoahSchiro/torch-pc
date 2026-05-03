# Based on concepts from Monadillo (MIT License)
# Source: https://github.com/Monadillo/pcn-intro/blob/main/pcn_cifar10_notebook.ipynb

from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import autocast


class PCLayer(nn.Module):
    """A single latent layer in a predictive coding network.

    Computes predictions of the layer below by applying a linear
    transformation followed by a pointwise nonlinearity:
        A^(l)      = X^(l+1) @ W^(l).T
        X_hat^(l)  = f^(l)(A^(l))

    Args:
        in_dim: Dimension of the layer above (d_{l+1}).
        out_dim: Dimension of this layer (d_l).
        activation_fn: Pointwise nonlinearity f^(l). Defaults to ReLU.
        activation_deriv: Elementwise derivative of activation_fn.
            Defaults to the subgradient of ReLU.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation_fn=torch.relu,
        activation_deriv=lambda a: (a > 0).float(),
    ):
        super().__init__()
        self.W = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.xavier_uniform_(self.W)
        self.activation_fn = activation_fn
        self.activation_deriv = activation_deriv

    def forward(self, x_above: torch.Tensor):
        """Return predictions and preactivations given the layer above.

        Args:
            x_above: Activations of the layer above, shape (B, d_{l+1}).

        Returns:
            x_hat: Predictions of this layer, shape (B, d_l).
            a: Preactivations, shape (B, d_l).
        """
        with autocast(device_type="cuda"):
            a = x_above @ self.W.T
            return self.activation_fn(a), a


class PCNetwork(nn.Module):
    """A fully-connected predictive coding network.

    The network consists of L latent layers plus a linear readout head.
    Predictions flow top-down: each layer predicts the activations of
    the layer below via a learned weight matrix and nonlinearity.

    Args:
        dims: List of layer dimensions [d_0, d_1, ..., d_L].
            d_0 is the input dimension; d_L is the top latent dimension.
        output_dim: Number of output units for the readout head.
        activation_fn: Nonlinearity applied in every PCLayer.
        activation_deriv: Elementwise derivative of activation_fn.
    """

    def __init__(
        self,
        dims: list[int],
        output_dim: int,
        activation_fn=torch.relu,
        activation_deriv=lambda a: (a > 0).float(),
    ):
        super().__init__()
        self.dims = dims
        self.L = len(dims) - 1
        self.layers = nn.ModuleList([
            PCLayer(
                in_dim=dims[l + 1],
                out_dim=dims[l],
                activation_fn=activation_fn,
                activation_deriv=activation_deriv,
            )
            for l in range(self.L)
        ])
        self.readout = nn.Linear(dims[-1], output_dim, bias=False)


    @staticmethod
    def load(fp: Path, device=None):

        if device:
            checkpoint = torch.load(fp, map_location=device)
        else:
            checkpoint = torch.load(fp)

        model = PCNetwork(
            dims=checkpoint["dims"],
            output_dim=checkpoint["output_dim"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        return model
    
    
    def save(self, fp: Path):
        torch.save({
            "dims": self.dims,
            "output_dim": self.readout.out_features,
            "state_dict": self.state_dict(),
        }, fp)

    
    def init_latents(self, batch_size: int, device: torch.device) -> list[torch.Tensor]:
        """Initialise latent variables X^(1), ..., X^(L) as standard normals.

        Args:
            batch_size: Number of samples in the batch.
            device: Device on which to allocate tensors.

        Returns:
            List of tensors of shapes [(B, d_1), ..., (B, d_L)].
        """
        return [
            torch.randn(batch_size, d, device=device, requires_grad=False)
            for d in self.dims[1:]
        ]

    def compute_errors(
        self, inputs_latents: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Compute prediction errors and gain-modulated errors for all layers.

        Args:
            inputs_latents: [X^(0), X^(1), ..., X^(L)], a list of tensors
                of shapes [(B, d_0), ..., (B, d_L)].

        Returns:
            errors: [E^(0), ..., E^(L-1)], prediction errors per layer.
            gain_modulated_errors: [H^(0), ..., H^(L-1)], errors scaled
                by the derivative of the activation function.
        """
        errors, gain_modulated_errors = [], []
        for l, layer in enumerate(self.layers):
            x_hat, a = layer(inputs_latents[l + 1])
            err = inputs_latents[l] - x_hat
            errors.append(err)
            gain_modulated_errors.append(err * layer.activation_deriv(a))
        return errors, gain_modulated_errors
