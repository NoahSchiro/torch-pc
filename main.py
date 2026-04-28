"""
CIFAR-10 classification with a Predictive Coding Network.

Trains a three-layer PCN on CIFAR-10 and reports Top-1 / Top-3 accuracy.
Energy trajectories are visualised interactively with Plotly.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from torch_pc import (
    PCNetwork,
    train_pcn,
    test_pcn,
)

# Hyperparameters
BATCH_SIZE = 500
num_epochs = 4
eta_infer  = 0.05
eta_learn  = 0.005
T_infer    = 50
T_learn    = BATCH_SIZE
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616),
    ),
])

trainset = torchvision.datasets.CIFAR10(root="./data", train=True,  download=True, transform=transform)
testset  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

loader_kwargs = dict(batch_size=BATCH_SIZE, num_workers=10, pin_memory=True, prefetch_factor=2)
trainloader = DataLoader(trainset, shuffle=True,  **loader_kwargs)
testloader  = DataLoader(testset,  shuffle=False, **loader_kwargs)

# Define model
model = PCNetwork(
    dims=[3072, 1000, 500, 10],
    output_dim=10,
)

# Train
print(f"Using device: {device}")
print("Starting PCN training...")

energy_history, supervised_energy_history = train_pcn(
    model=model,
    data_loader=trainloader,
    num_epochs=num_epochs,
    eta_infer=eta_infer,
    eta_learn=eta_learn,
    T_infer=T_infer,
    T_learn=T_learn,
    device=device,
)

print("Training finished.")

# Evaluate
acc1, acc3 = test_pcn(
    model=model,
    data_loader=testloader,
    T_infer=T_infer,
    eta_infer=eta_infer,
    device=device,
)
print(f"Test Top-1 Accuracy: {acc1 * 100:.2f}%")
print(f"Test Top-3 Accuracy: {acc3 * 100:.2f}%")

# Plot

# plot_energy_history(energy_history, T_infer, T_learn)
# plot_energy_history(
#     supervised_energy_history, T_infer, T_learn,
#     title="Batch-Averaged Supervised Energy Trajectories",
# )
# plot_epoch_avg_energy(energy_history, T_infer, T_learn)
# plot_epoch_avg_energy(
#     supervised_energy_history, T_infer, T_learn,
#     title="Batch-Averaged Supervised Energy Trajectories (Mean ± 1-std)",
# )
