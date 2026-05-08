"""
CIFAR-10 classification with a Predictive Coding Network.

Trains a three-layer PCN on CIFAR-10 and reports Top-1 / Top-3 accuracy.
Energy trajectories are visualised interactively with Plotly.
"""

import os

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from torch_pc import (
    PCNetwork,
    train_pcn,
    test_pcn_classify,
)

# Hyperparameters
BATCH_SIZE  = 500
EPOCHS      = 4
ETA_INFER   = 0.05
ETA_LEARN   = 0.005
INFER_STEPS = 50
T_LEARN     = BATCH_SIZE
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616),
    ),
])

trainset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)
testset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

trainloader = DataLoader(
    trainset,
    shuffle=True,
    batch_size=BATCH_SIZE,
    num_workers=os.cpu_count(),
    pin_memory=True,
    prefetch_factor=2
)
testloader = DataLoader(
    testset,
    shuffle=False,
    batch_size=BATCH_SIZE,
    num_workers=os.cpu_count(),
    pin_memory=True,
    prefetch_factor=2
)

# Define model
model = PCNetwork(
    dims=[3072, 1000, 500, 10],
    output_dim=10,
)

# Train
print(f"Using device: {DEVICE}")
print("Starting PCN training...")

energy_history, supervised_energy_history = train_pcn(
    model=model,
    data_loader=trainloader,
    num_epochs=EPOCHS,
    eta_infer=ETA_INFER,
    eta_learn=ETA_LEARN,
    infer_steps=INFER_STEPS,
    T_learn=T_LEARN,
    device=DEVICE,
)

print("Training finished.")

# Evaluate
acc1, acc3 = test_pcn_classify(
    model=model,
    data_loader=testloader,
    infer_steps=INFER_STEPS,
    eta_infer=ETA_INFER,
    device=DEVICE,
)
print(f"Test Top-1 Accuracy: {acc1 * 100:.2f}%")
print(f"Test Top-3 Accuracy: {acc3 * 100:.2f}%")
