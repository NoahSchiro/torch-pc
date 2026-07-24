"""
CIFAR-10 classification with a Predictive Coding Network.

Trains a three-layer PCN on CIFAR-10 and reports Top-1 / Top-3 accuracy.
Energy trajectories are visualised interactively with Plotly.

Experimentally, the hyperparameters below have gotten me:
    Top-1: 29.6%
    Top-3: 59.0%
I did not do too much hyperparameter tuning for this one
"""

import os
import random

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from torch_pc import (
    PCNetwork,
    train_pcn,
    test_pcn_classify,
)

SEED = 64

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Hyperparameters
BATCH_SIZE        = 512
EPOCHS            = 2
ETA_INFER         = 0.05
ETA_LEARN         = 0.005
TRAIN_INFER_STEPS = 50
TEST_INFER_STEPS  = 300
T_LEARN           = BATCH_SIZE
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get data. Normalization stats come from torchvision
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
    dims=[3072, 1024, 512, 128],
    output_dim=10,
    activation_fn=torch.tanh,
    activation_deriv=lambda a: 1 - torch.tanh(a)**2,
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
    infer_steps=TRAIN_INFER_STEPS,
    T_learn=T_LEARN,
    device=DEVICE,
)

print("Training finished.")

# Evaluate
acc1, acc3 = test_pcn_classify(
    model=model,
    data_loader=testloader,
    infer_steps=TEST_INFER_STEPS,
    eta_infer=ETA_INFER,
    device=DEVICE,
)
print(f"Test Top-1 Accuracy: {acc1 * 100:.2f}%")
print(f"Test Top-3 Accuracy: {acc3 * 100:.2f}%")
