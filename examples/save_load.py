"""
Same as `cifar10.py` but demos saving and loading

Trains a three-layer PCN on CIFAR-10 and reports Top-1 / Top-3 accuracy.
Energy trajectories are visualised interactively with Plotly.
"""

import os
from argparse import ArgumentParser
from pathlib import Path

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

def save(fp: Path):
    trainset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
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
    model = PCNetwork(
        dims=[3072, 1000, 500, 10],
        output_dim=10,
    )

    print(f"Using device: {device}")
    print("Starting PCN training...")

    _ = train_pcn(
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

    torch.save({
        "dims": model.dims,
        "output_dim": model.readout.out_features,
        "state_dict": model.state_dict(),
    }, fp)
    print("Model saved")

def load(fp: Path):
    testset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )
    testloader = DataLoader(
        testset,
        shuffle=False,
        batch_size=BATCH_SIZE,
        num_workers=os.cpu_count(),
        pin_memory=True,
        prefetch_factor=2
    )
    checkpoint = torch.load(fp, map_location=device)
    model = PCNetwork(
        dims=checkpoint["dims"],
        output_dim=checkpoint["output_dim"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    acc1, acc3 = test_pcn(
        model=model,
        data_loader=testloader,
        T_infer=T_infer,
        eta_infer=eta_infer,
        device=device,
    )
    print(f"Test Top-1 Accuracy: {acc1 * 100:.2f}%")
    print(f"Test Top-3 Accuracy: {acc3 * 100:.2f}%")

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", required=True, type=str)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--load", action="store_true")
    args = parser.parse_args()

    fp = Path(args.path)
    if args.save:
        save(fp)
    else:
        load(fp)
