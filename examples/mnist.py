"""
Classic MNIST but with Predictive Coding Networks
"""

import os

import torch
from torch.utils.data import DataLoader 
import torchvision
import torchvision.transforms as transforms

from torch_pc import PCNetwork, train_pcn, test_pcn

# best top-1: 99.95%
# best top-3: 100%

BATCH_SIZE  = 256  # best: 256
EPOCHS      = 1    # best: 2, though 1 can achieve 99.88
ETA_INFER   = 0.1  # best: 0.1
ETA_LEARN   = 0.02 # best: 0.02
INFER_STEPS = 80   # best: 80
T_LEARN     = 3    # best: 3
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Define model
model = PCNetwork(
    # best: [784, 1000, 500, 100]
    dims=[784, 1000, 500, 100],
    output_dim=10,
)

# Train
print(f"Using device: {DEVICE}")
print("Starting PCN training...")
energy_history, supervised_energy_history = train_pcn(
    model=model,
    data_loader=train_dl,
    num_epochs=EPOCHS,
    eta_infer=ETA_INFER,
    eta_learn=ETA_LEARN,
    infer_steps=INFER_STEPS,
    T_learn=T_LEARN,
    device=DEVICE,
)
print("Training finished.")

# Evaluate
acc1, acc3 = test_pcn(
    model=model,
    data_loader=test_dl,
    infer_steps=INFER_STEPS,
    eta_infer=ETA_INFER,
    device=DEVICE,
)
print(f"Test Top-1 Accuracy: {acc1 * 100:.2f}%")
print(f"Test Top-3 Accuracy: {acc3 * 100:.2f}%")
