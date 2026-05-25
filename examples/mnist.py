"""
Classic MNIST but with Predictive Coding Networks

With a bit of hyperparameter tuning:
  Top-1: 63.86%
  Top-3: 84.82%
"""

import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader 
import torchvision
import torchvision.transforms as transforms

from torch_pc import PCNetwork, train_pcn, test_pcn_classify

SEED = 64

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

BATCH_SIZE         = 256
EPOCHS             = 3
ETA_TRAIN_INFER_LR = 0.1
ETA_TEST_INFER_LR  = 0.1
ETA_LEARN          = 0.02
TRAIN_INFER_STEPS  = 80
TEST_INFER_STEPS   = 300
T_LEARN            = 3
DEVICE             = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.1307,),
        (0.3081,),
    ),
])
train_ds = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)
test_ds  = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_dl = DataLoader(
    train_ds,
    shuffle=True,
    batch_size=BATCH_SIZE,
    num_workers=os.cpu_count(),
    pin_memory=True,
)
test_dl = DataLoader(
    test_ds,
    shuffle=False,
    batch_size=BATCH_SIZE,
    num_workers=os.cpu_count(),
    pin_memory=True,
    prefetch_factor=2
)

# Model
model = PCNetwork(
    # best: [784, 1000, 500, 100]
    dims=[784, 1000, 500, 100],
    output_dim=10,
    activation_fn=torch.tanh,
    activation_deriv=lambda a: 1 - torch.tanh(a)**2,
)

# Train
print(f"Using device: {DEVICE}")
print("Starting PCN training...")
energy_history, supervised_energy_history = train_pcn(
    model=model,
    data_loader=train_dl,
    num_epochs=EPOCHS,
    eta_infer=ETA_TRAIN_INFER_LR,
    eta_learn=ETA_LEARN,
    infer_steps=TRAIN_INFER_STEPS,
    T_learn=T_LEARN,
    device=DEVICE,
)
print("Training finished.")

# Evaluate
acc1, acc3 = test_pcn_classify(
    model=model,
    data_loader=test_dl,
    infer_steps=TEST_INFER_STEPS,
    eta_infer=ETA_TEST_INFER_LR,
    device=DEVICE,
)
print(f"Test Top-1 Accuracy: {acc1 * 100:.2f}%")
print(f"Test Top-3 Accuracy: {acc3 * 100:.2f}%")
