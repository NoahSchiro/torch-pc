# Based on concepts from Monadillo (MIT License)
# Source: https://github.com/Monadillo/pcn-intro/blob/main/pcn_cifar10_notebook.ipynb

import warnings
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.amp import autocast


class PCLayer(nn.Module):
    """A single latent layer in a predictive coding network.

    Computes predictions of the layer below by applying a linear
    transformation followed by a pointwise nonlinearity:
        A^(l)      = X^(l+1) @ W^(l).T  [+ b^(l)]
        X_hat^(l)  = f^(l)(A^(l))

    Args:
        in_dim: Dimension of the layer above (d_{l+1}).
        out_dim: Dimension of this layer (d_l).
        activation_fn: Pointwise nonlinearity f^(l). Defaults to ReLU.
        activation_deriv: Elementwise derivative of activation_fn.
            Defaults to the subgradient of ReLU.
        use_bias: If True, adds a learnable bias vector b^(l) of shape
            (d_l,) to the preactivation. Defaults to False.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation_fn: Callable = torch.relu,
        activation_deriv: Callable = lambda a: (a > 0).float(),
        use_bias: bool = False,
    ):
        super().__init__()
        self.W = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.xavier_uniform_(self.W)
        self.bias = nn.Parameter(torch.empty(out_dim).normal_(std=0.01)) if use_bias else None
        
        self.activation_fn = activation_fn
        self.activation_deriv = activation_deriv

    @property
    def use_bias(self) -> bool:
        return self.bias is not None

    def forward(self, x_above: torch.Tensor):
        """Return predictions and preactivations given the layer above.

        Args:
            x_above: Activations of the layer above, shape (B, d_{l+1}).

        Returns:
            x_hat: Predictions of this layer, shape (B, d_l).
            a: Preactivations before the nonlinearity, shape (B, d_l).
        """
        with autocast(device_type="cuda"):
            a = x_above @ self.W.T
            if self.bias is not None:
                a = a + self.bias
            return self.activation_fn(a), a


class PCNetwork(nn.Module):
    """A fully-connected predictive coding network.

    The network consists of L latent layers plus a linear readout head.
    Predictions flow top-down: each layer predicts the activations of
    the layer below via a learned weight matrix, optional bias, and
    nonlinearity.

    Args:
        dims: List of layer dimensions [d_0, d_1, ..., d_L].
            d_0 is the input dimension; d_L is the top latent dimension.
        output_dim: Number of output units for the readout head.
        activation_fn: Nonlinearity applied in every PCLayer.
        activation_deriv: Elementwise derivative of activation_fn.
        use_bias: If True, every PCLayer and the readout head include a
            learnable bias term. Defaults to False.
    """

    def __init__(
        self,
        dims: list[int],
        output_dim: int,
        activation_fn: Callable = torch.relu,
        activation_deriv: Callable = lambda a: (a > 0).float(),
        use_bias: bool = False,
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
                use_bias=use_bias,
            )
            for l in range(self.L)
        ])
        self.readout = nn.Linear(dims[-1], output_dim, bias=use_bias)

    
    @staticmethod
    def load(fp: Path, device=None):
        """Load a PCNetwork from a checkpoint file saved with `save`.

        Args:
            fp: Path to the checkpoint file.
            device: Optional device to map tensors onto.

        Returns:
            The reconstructed PCNetwork with weights restored.
        """
        checkpoint = torch.load(fp, map_location=device)
        model = PCNetwork(
            dims=checkpoint["dims"],
            output_dim=checkpoint["output_dim"],
            use_bias=checkpoint.get("use_bias", False),
        )
        model.load_state_dict(checkpoint["state_dict"])
        return model

    def save(self, fp: Path):
        """Save this network to a checkpoint file.

        Args:
            fp: Destination path for the checkpoint.
        """
        torch.save({
            "dims": self.dims,
            "output_dim": self.readout.out_features,
            "use_bias": self.layers[0].use_bias if self.layers else False,
            "state_dict": self.state_dict(),
        }, fp)

    @classmethod
    def from_bp(
        cls,
        seq: nn.Sequential,
        activation_fn: Callable = torch.relu,
        activation_deriv: Callable = lambda a: (a > 0).float(),
    ) -> "PCNetwork":
        """Construct a PCNetwork from a standard backprop `nn.Sequential`.

        The Sequential is expected to alternate `nn.Linear` layers with
        pointwise activation modules, ending with a final `nn.Linear`
        readout (no trailing activation).  Activation modules are skipped
        during parsing. Only the linear weight/bias values are transferred.
        Provide matching `activation_fn` / `activation_deriv` callables
        so the PC inference loop uses the correct nonlinearity.

        Example input:

            nn.Sequential(
                nn.Linear(784, 512),  nn.ReLU(),
                nn.Linear(512, 256),  nn.ReLU(),
                nn.Linear(256, 128),  nn.ReLU(),
                nn.Linear(128, 10),
            )

        Args:
            seq: A trained (or randomly initialised) `nn.Sequential`.
            activation_fn: The nonlinearity used in the Sequential's hidden
                layers.  This is passed to every PCLayer.
            activation_deriv: Elementwise derivative of `activation_fn`.

        Returns:
            A PCNetwork whose weights (and biases, if present) are copied
            from `seq`.

        Raises:
            ValueError: If fewer than two `nn.Linear` layers are found, or
                if the last linear layer is followed by an activation.
        """
        linear_layers: list[nn.Linear] = [
            m for m in seq.modules() if isinstance(m, nn.Linear)
        ]
        if len(linear_layers) < 2:
            raise ValueError(
                "Sequential must contain at least two nn.Linear layers "
                "(one or more hidden + one readout)."
            )

        # Verify the last module is a Linear.
        last_non_container = [
            m for m in seq.children()
            if not isinstance(m, nn.Sequential)
        ][-1]
        if not isinstance(last_non_container, nn.Linear):
            raise ValueError(
                "The last module in the Sequential must be nn.Linear "
                "(the readout). Remove any trailing activation."
            )

        hidden_linears = linear_layers[:-1]
        readout_linear = linear_layers[-1]

        # Detect bias usage; warn if layers are inconsistent.
        bias_flags = [lin.bias is not None for lin in linear_layers]
        use_bias = bias_flags[0]
        if len(set(bias_flags)) > 1:
            warnings.warn(
                "Not all Linear layers have the same bias setting. "
                "Using the first layer's setting "
                f"(use_bias={use_bias}) for all PCLayers.",
                UserWarning,
                stacklevel=2,
            )

        # Build dims: [d_0, d_1, ..., d_L]
        dims = [hidden_linears[0].in_features] + [
            lin.out_features for lin in hidden_linears
        ]
        output_dim = readout_linear.out_features

        pc_net = cls(
            dims=dims,
            output_dim=output_dim,
            activation_fn=activation_fn,
            activation_deriv=activation_deriv,
            use_bias=use_bias,
        )

        if use_bias:
            warnings.warn(
                "Biases cannot be transferred from a BP network to a PCNetwork",
                UserWarning,
                stacklevel=2,
            )

        with torch.no_grad():
            for pc_layer, bp_layer in zip(pc_net.layers, hidden_linears):
                # bp_layer.weight: (d_{l+1}, d_l)  →  pc_layer.W: (d_l, d_{l+1})
                pc_layer.W.copy_(bp_layer.weight.T)
                # PC biases stay at zero (see note above).

            # Readout: both nn.Linear, same (output_dim, d_L) shape — copy directly.
            pc_net.readout.weight.copy_(readout_linear.weight)
            if use_bias and pc_net.readout.bias is not None and readout_linear.bias is not None:
                pc_net.readout.bias.copy_(readout_linear.bias)

        return pc_net

    def to_bp(self, activation_module: nn.Module | None = None) -> nn.Sequential:
        """Export this PCNetwork as a standard backprop `nn.Sequential`.

        Each PCLayer becomes an `nn.Linear` followed by an activation
        module.  The readout becomes a final `nn.Linear` with no trailing
        activation.

        Args:
            activation_module: An instantiated `nn.Module` used as the
                activation after each hidden linear layer, e.g.
                `nn.ReLU()`, `nn.Tanh()`.  A *new instance* is inserted
                after every hidden layer so each has independent state
                (important for modules like `nn.PReLU`).  Defaults to
                `nn.ReLU()` if not provided.

        Returns:
            An `nn.Sequential` whose weights (and biases, if this network
            uses them) are copied from this PCNetwork.  The returned module
            is in the same training/eval state as this network.
        """
        if activation_module is None:
            activation_module = nn.ReLU()

        use_bias = self.layers[0].use_bias if self.layers else False

        if use_bias:
            warnings.warn(
                "Biases cannot be transferred from a PCNetwork to an nn.Sequential",
                UserWarning,
                stacklevel=2,
            )
        layers_list: list[nn.Module] = []

        with torch.no_grad():
            for pc_layer in self.layers:
                # pc_layer.W shape: (d_l, d_{l+1})
                # BP Linear maps bottom-up d_l -> d_{l+1}:
                #   in_features = d_l, out_features = d_{l+1}
                #   weight shape = (d_{l+1}, d_l) = pc_layer.W.T
                in_f  = pc_layer.W.shape[0]  # d_l
                out_f = pc_layer.W.shape[1]  # d_{l+1}
                lin = nn.Linear(in_f, out_f, bias=use_bias)
                lin.weight.copy_(pc_layer.W.T)
                # PC biases (shape d_l) shift top-down predictions; BP biases
                # (shape d_{l+1}) shift bottom-up pre-activations.  They have
                # different shapes and semantics, so BP biases are zeroed here.
                # The caller should retrain the BP biases after export.
                if use_bias and lin.bias is not None:
                    nn.init.zeros_(lin.bias)
                layers_list.append(lin)
                # Instantiate a fresh copy so each layer has independent params.
                layers_list.append(type(activation_module)())

            readout = nn.Linear(
                self.readout.in_features,
                self.readout.out_features,
                bias=use_bias,
            )
            readout.weight.copy_(self.readout.weight)
            if use_bias and self.readout.bias is not None:
                readout.bias.copy_(self.readout.bias)
            layers_list.append(readout)

        seq = nn.Sequential(*layers_list)
        seq.train(self.training)
        return seq

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
