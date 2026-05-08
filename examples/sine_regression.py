"""
Sine wave regression with Predictive Coding Networks.

Given a scalar input x sampled uniformly from [0, 2π],
predict sin(x).
"""
import os

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch_pc import PCNetwork, train_pcn, test_pcn_regress

# Hyperparameters
BATCH_SIZE  = 256
EPOCHS      = 10
ETA_INFER   = 0.1
ETA_LEARN   = 0.02
INFER_STEPS = 80
T_LEARN     = 3
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_dataset(n: int) -> TensorDataset:
    x = torch.FloatTensor(n).uniform_(0, 2 * torch.pi)
    x = x.unsqueeze(1)
    y = torch.sin(x)
    return TensorDataset(x, y)

train_ds = make_dataset(50_000)
test_ds  = make_dataset(10_000)

train_dl = DataLoader(train_ds, shuffle=True,  batch_size=BATCH_SIZE, num_workers=os.cpu_count(), pin_memory=True)
test_dl  = DataLoader(test_ds,  shuffle=False, batch_size=BATCH_SIZE, num_workers=os.cpu_count(), pin_memory=True)

# Model
model = PCNetwork(
    dims=[1, 128, 64, 32],
    output_dim=1,
)

# Train
print(f"Using device: {DEVICE}")
print("Starting PCN sine regression training...")

energy_history, supervised_energy_history = train_pcn(
    model=model,
    data_loader=train_dl,
    num_epochs=EPOCHS,
    eta_infer=ETA_INFER,
    eta_learn=ETA_LEARN,
    infer_steps=INFER_STEPS,
    T_learn=T_LEARN,
    device=DEVICE,
    target_transform=lambda y: y.float().unsqueeze(1),
)

print("Training finished.")

# Evaluate
mse, mae = test_pcn_regress(
    model=model,
    data_loader=test_dl,
    infer_steps=INFER_STEPS,
    eta_infer=ETA_INFER,
    device=DEVICE,
)

print(f"Test MSE: {mse:.6f}")
print(f"Test MAE: {mae:.6f}")
