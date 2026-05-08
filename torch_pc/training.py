# Based on concepts from Monadillo (MIT License)
# Source: https://github.com/Monadillo/pcn-intro/blob/main/pcn_cifar10_notebook.ipynb
from typing import Callable

import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from .module import PCNetwork


def train_pcn(
    model: PCNetwork,
    data_loader: DataLoader,
    num_epochs: int,
    eta_infer: float,
    eta_learn: float,
    infer_steps: int,
    T_learn: int,
    target_transform: Callable | None = None,
    device: str | torch.device = "cuda",
) -> tuple[list, list]:
    """Train a PCNetwork on a supervised task.
 
    Each batch proceeds in two phases:
      1. Inference: update latent variables X^(1), ..., X^(L) for
         infer_steps while weights and X^(0) remain fixed.
      2. Learning: update all weights for T_learn steps while latents
         remain fixed at their inferred values.
 
    For classification, labels are converted to one-hot vectors by default.
    For regression, pass ``target_transform=None`` and supply continuous
    targets directly from the DataLoader (or provide a custom transform).
 
    Args:
        model: The PCNetwork to train.
        data_loader: Yields (input, target) batches.
        num_epochs: Number of full passes over the dataset.
        eta_infer: Step size for latent variable updates.
        eta_learn: Step size for weight updates.
        infer_steps: Number of inference steps per batch.
        T_learn: Number of learning steps per batch.
        device: Device to run training on.
        target_transform: Optional callable applied to the raw target batch
            before computing the supervised error. Defaults to one-hot
            encoding with ``model.readout.out_features`` classes, which is
            the correct behaviour for classification. Pass the identity
            (``lambda y: y``) or any other transform for regression.
 
    Returns:
        energy_history: Nested list [epoch][batch][step] of batch-averaged
            total energy values (latent + supervised).
        supervised_energy_history: Same structure, supervised energy only.
    """
    model.to(device).train()
    energy_history, supervised_energy_history = [], []

    if target_transform is None:
        num_classes = model.readout.out_features
        target_transform = lambda y: F.one_hot(y, num_classes=num_classes).float()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1} / {num_epochs}")
        epoch_energies, epoch_supervised_energies = [], []

        for x_batch, y_batch in tqdm(data_loader):
            B = x_batch.size(0)
            x_batch = x_batch.view(B, model.dims[0]).to(device)
            y_batch = target_transform(y_batch).to(device)

            inputs_latents = [x_batch] + model.init_latents(B, device)
            weights = [layer.W for layer in model.layers] + [model.readout.weight]

            batch_energies, batch_supervised_energies = [], []

            # Initial predictions before any updates (t = 0)
            errors, gain_modulated_errors = model.compute_errors(inputs_latents)
            y_hat = model.readout(inputs_latents[-1])
            eps_sup = y_hat - y_batch
            eps_L = eps_sup @ weights[-1]
            errors_extended = errors + [eps_L]

            batch_supervised_energies.append(0.5 * eps_sup.pow(2).sum().item() / B)
            batch_energies.append(
                0.5 * sum(e.pow(2).sum().item() for e in errors) / B
                + batch_supervised_energies[-1]
            )

            # Inference phase
            with torch.no_grad(), autocast(device_type="cuda"):
                for _ in range(infer_steps):
                    for l in range(1, model.L + 1):
                        grad_Xl = errors_extended[l] - gain_modulated_errors[l - 1] @ weights[l - 1]
                        inputs_latents[l] -= eta_infer * grad_Xl

                    errors, gain_modulated_errors = model.compute_errors(inputs_latents)
                    y_hat = model.readout(inputs_latents[-1])
                    eps_sup = y_hat - y_batch
                    eps_L = eps_sup @ weights[-1]
                    errors_extended = errors + [eps_L]

                    batch_supervised_energies.append(0.5 * eps_sup.pow(2).sum().item() / B)
                    batch_energies.append(
                        0.5 * sum(e.pow(2).sum().item() for e in errors) / B
                        + batch_supervised_energies[-1]
                    )

            # Learning phase
            with torch.no_grad():
                for _ in range(T_learn):
                    for l in range(model.L):
                        grad_Wl = -(gain_modulated_errors[l].T @ inputs_latents[l + 1]) / B
                        weights[l] -= eta_learn * grad_Wl
                    grad_Wout = eps_sup.T @ inputs_latents[-1] / B
                    weights[-1] -= eta_learn * grad_Wout

                    errors, gain_modulated_errors = model.compute_errors(inputs_latents)
                    y_hat = model.readout(inputs_latents[-1])
                    eps_sup = y_hat - y_batch

                    sup_e = 0.5 * eps_sup.pow(2).sum().item() / B
                    lat_e = 0.5 * sum(e.pow(2).sum().item() for e in errors) / B
                    batch_supervised_energies.append(sup_e)
                    batch_energies.append(lat_e + sup_e)

            epoch_energies.append(batch_energies)
            epoch_supervised_energies.append(batch_supervised_energies)

        energy_history.append(epoch_energies)
        supervised_energy_history.append(epoch_supervised_energies)

    return energy_history, supervised_energy_history

@torch.no_grad()
def _run_inference(
    model: PCNetwork,
    x_batch: torch.Tensor,
    y_onehot: torch.Tensor,
    infer_steps: int,
    eta_infer: float,
) -> list[torch.Tensor]:
    """Run the latent inference loop for a single batch.
 
    Iteratively minimises the free energy with respect to the latent
    variables X^(1), ..., X^(L) while keeping weights fixed. Both
    classification and regression evaluation share this procedure.
 
    Args:
        model: A PCNetwork in eval mode.
        x_batch: Input activations, shape (B, d_0). Already on device.
        y_onehot: Supervised targets, shape (B, output_dim). Already on
            device. For classification these are one-hot vectors; for
            regression they are raw continuous targets.
        infer_steps: Number of gradient steps on the latents.
        eta_infer: Step size for latent updates.
 
    Returns:
        inputs_latents: [X^(0), X^(1), ..., X^(L)] after inference,
            shapes [(B, d_0), ..., (B, d_L)].
    """
    B = x_batch.size(0)
    inputs_latents = [x_batch] + model.init_latents(B, x_batch.device)
    weights = [layer.W for layer in model.layers] + [model.readout.weight]
 
    with autocast(device_type="cuda"):
        for _ in range(infer_steps):
            errors, gain_modulated_errors = model.compute_errors(inputs_latents)
            y_hat = model.readout(inputs_latents[-1])
            eps_sup = y_hat - y_onehot
            eps_L = eps_sup @ weights[-1]
            errors_extended = errors + [eps_L]
 
            for l in range(1, model.L + 1):
                grad_Xl = errors_extended[l] - gain_modulated_errors[l - 1] @ weights[l - 1]
                inputs_latents[l] -= eta_infer * grad_Xl
 
    return inputs_latents
 
 
@torch.no_grad()
def test_pcn_classify(
    model: PCNetwork,
    data_loader: DataLoader,
    infer_steps: int,
    eta_infer: float,
    device: str | torch.device = "cuda",
) -> tuple[float, float]:
    """Evaluate a PCNetwork on a classification dataset.
 
    Runs the inference procedure then computes Top-1 and Top-3 accuracy
    from the readout layer.
 
    Args:
        model: Trained PCNetwork.
        data_loader: Yields (input, label) batches where labels are integer
            class indices.
        infer_steps: Number of inference steps per batch.
        eta_infer: Step size for latent variable updates.
        device: Device to run evaluation on.
 
    Returns:
        top1_acc: Top-1 accuracy in [0, 1].
        top3_acc: Top-3 accuracy in [0, 1].
    """
    model.to(device).eval()
    total, top1_correct, top3_correct = 0, 0, 0
 
    for x_batch, y_batch in tqdm(data_loader):
        B = x_batch.size(0)
        total += B
        x_batch = x_batch.view(B, model.dims[0]).to(device)
        y_labels = y_batch.to(device)
        y_onehot = F.one_hot(y_labels, num_classes=model.readout.out_features).float()
 
        inputs_latents = _run_inference(model, x_batch, y_onehot, infer_steps, eta_infer)
 
        logits = model.readout(inputs_latents[-1])
        top1_correct += (logits.argmax(dim=1) == y_labels).sum().item()
        _, preds3 = logits.topk(3, dim=1)
        top3_correct += (preds3 == y_labels.unsqueeze(1)).any(dim=1).sum().item()
 
    return top1_correct / total, top3_correct / total
 
 
@torch.no_grad()
def test_pcn_regress(
    model: PCNetwork,
    data_loader: DataLoader,
    infer_steps: int,
    eta_infer: float,
    device: str | torch.device = "cuda",
) -> tuple[float, float]:
    """Evaluate a PCNetwork on a regression dataset.
 
    Runs the same latent inference procedure as classification evaluation,
    then computes MSE and MAE between the readout predictions and the
    continuous targets.
 
    Args:
        model: Trained PCNetwork.
        data_loader: Yields (input, target) batches where targets are
            continuous float tensors of shape (B,) or (B, output_dim).
        infer_steps: Number of inference steps per batch.
        eta_infer: Step size for latent variable updates.
        device: Device to run evaluation on.
 
    Returns:
        mse: Mean squared error across the full dataset.
        mae: Mean absolute error across the full dataset.
    """
    model.to(device).eval()
    total = 0
    sum_sq_err = 0.0
    sum_abs_err = 0.0
 
    for x_batch, y_batch in tqdm(data_loader):
        B = x_batch.size(0)
        total += B
        x_batch = x_batch.view(B, model.dims[0]).to(device)
        y_batch = y_batch.float().to(device)
 
        # Ensure targets are (B, output_dim) for a consistent supervised signal
        if y_batch.dim() == 1:
            y_batch = y_batch.unsqueeze(1)
 
        inputs_latents = _run_inference(model, x_batch, y_batch, infer_steps, eta_infer)
 
        preds = model.readout(inputs_latents[-1])
        residuals = preds - y_batch
        sum_sq_err += residuals.pow(2).sum().item()
        sum_abs_err += residuals.abs().sum().item()
 
    # Normalise over total number of (sample, output) pairs
    n = total * model.readout.out_features
    return sum_sq_err / n, sum_abs_err / n
