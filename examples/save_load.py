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
    test_pcn_classify,
)

BATCH_SIZE  = 256
EPOCHS      = 2
ETA_INFER   = 0.1
ETA_LEARN   = 0.02
INFER_STEPS = 80
T_LEARN     = 3
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.1307,),
        (0.3081,),
    ),
])

def save(fp: Path):
    train_ds = torchvision.datasets.MNIST(
        root="./data",
        train=True,
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
    model = PCNetwork(
        dims=[784, 1000, 500, 100],
        output_dim=10,
    )

    # Train
    print(f"Using device: {DEVICE}")
    print("Starting PCN training...")
    _ = train_pcn(
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

    model.save(fp)
    print("Model saved")

def load(fp: Path):
    test_ds  = torchvision.datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )
    test_dl = DataLoader(
        test_ds,
        shuffle=False,
        batch_size=BATCH_SIZE,
        num_workers=os.cpu_count(),
        pin_memory=True,
        prefetch_factor=2
    )
    model = PCNetwork.load(fp, DEVICE)
    acc1, acc3 = test_pcn_classify(
        model=model,
        data_loader=test_dl,
        infer_steps=INFER_STEPS,
        eta_infer=ETA_INFER,
        device=DEVICE,
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
