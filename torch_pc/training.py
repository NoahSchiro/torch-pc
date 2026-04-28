# Based on concepts from Monadillo (MIT License)
# Source: https://github.com/Monadillo/pcn-intro/blob/main/pcn_cifar10_notebook.ipynb
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
    T_infer: int,
    T_learn: int,
    device: str | torch.device = "cuda",
) -> tuple[list, list]:
    """Train a PCNetwork on a supervised classification task.

    Each batch proceeds in two phases:
      1. Inference: update latent variables X^(1), ..., X^(L) for T_infer
         steps while weights and X^(0) remain fixed.
      2. Learning: update all weights for T_learn steps while latents
         remain fixed at their inferred values.

    Labels are converted to one-hot vectors and a supervised prediction
    error is computed via a linear readout head.

    Args:
        model: The PCNetwork to train.
        data_loader: Yields (input, label) batches.
        num_epochs: Number of full passes over the dataset.
        eta_infer: Step size for latent variable updates.
        eta_learn: Step size for weight updates.
        T_infer: Number of inference steps per batch.
        T_learn: Number of learning steps per batch.
        device: Device to run training on.

    Returns:
        energy_history: Nested list [epoch][batch][step] of batch-averaged
            total energy values (latent + supervised).
        supervised_energy_history: Same structure, supervised energy only.
    """
    model.to(device).train()
    energy_history, supervised_energy_history = [], []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1} / {num_epochs}")
        epoch_energies, epoch_supervised_energies = [], []

        for x_batch, y_batch in tqdm(data_loader):
            B = x_batch.size(0)
            x_batch = x_batch.view(B, model.dims[0]).to(device)
            y_batch = F.one_hot(y_batch, num_classes=model.readout.out_features).float().to(device)

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
                for _ in range(T_infer):
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
def test_pcn(
    model: PCNetwork,
    data_loader: DataLoader,
    T_infer: int,
    eta_infer: float,
    device: str | torch.device = "cuda",
) -> tuple[float, float]:
    """Evaluate a PCNetwork on a classification dataset.

    Runs the same inference procedure as training (without weight updates)
    then computes Top-1 and Top-3 accuracy from the readout layer.

    Args:
        model: Trained PCNetwork.
        data_loader: Yields (input, label) batches.
        T_infer: Number of inference steps per batch.
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
        y_onehot = F.one_hot(y_labels, num_classes=model.readout.out_features).float().to(device)

        inputs_latents = [x_batch] + model.init_latents(B, device)
        weights = [layer.W for layer in model.layers] + [model.readout.weight]

        with autocast(device_type="cuda"):
            for _ in range(T_infer):
                errors, gain_modulated_errors = model.compute_errors(inputs_latents)
                y_hat = model.readout(inputs_latents[-1])
                eps_sup = y_hat - y_onehot
                eps_L = eps_sup @ weights[-1]
                errors_extended = errors + [eps_L]

                for l in range(1, model.L + 1):
                    grad_Xl = errors_extended[l] - gain_modulated_errors[l - 1] @ weights[l - 1]
                    inputs_latents[l] -= eta_infer * grad_Xl

        logits = model.readout(inputs_latents[-1])
        top1_correct += (logits.argmax(dim=1) == y_labels).sum().item()
        _, preds3 = logits.topk(3, dim=1)
        top3_correct += (preds3 == y_labels.unsqueeze(1)).any(dim=1).sum().item()

    return top1_correct / total, top3_correct / total
