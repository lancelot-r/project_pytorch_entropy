#!/usr/bin/env python
"""
Usage example
-------------
export PYTORCH_ENABLE_MPS_FALLBACK=1

python code/metastat_main.py \           
    --config config/config_train.json
"""

import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from metastat_dataloader import MetaStatDataset, collate_fn, load_data
from metastat_model import MyTransformerEstimator
import numpy as np
import os
import json
from tqdm import tqdm

def train_epoch(model, loader, optimizer, loss_fn, device, accum_steps=4):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, (x, lengths, y) in enumerate(loader):
        x, lengths, y = x.to(device), lengths.to(device), y.to(device)

        pred = model(x, lengths)
        loss = loss_fn(pred, y) / accum_steps
        loss.backward()

        if (step + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps  # undo scaling for logging

    # Handle leftover gradients
    if (step + 1) % accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / len(loader)


@torch.inference_mode()
def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    for x, lengths, y in loader:
        x, lengths, y = x.to(device), lengths.to(device), y.to(device)
        pred = model(x, lengths)
        loss = loss_fn(pred, y)
        total_loss += loss.item()
    return total_loss / len(loader)




def main():
    parser = argparse.ArgumentParser(description="Meta-Statistical Entropy Estimator Training (JSON version)")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file '{args.config}' not found.")
    
    with open(args.config, "r") as f:
        cfg = json.load(f)
    print(f"Loaded training configuration from {args.config}")

    # Extract parameters from JSON
    name = cfg["name"]
    data_name = cfg["data"]
    data_path = os.path.join("data", data_name, f"{data_name}.npz")
    val_name = cfg.get("val_data")
    if val_name is not None:
        val_path = os.path.join("data", val_name, f"{val_name}.npz")
    else:
        val_path = None
    epochs = cfg["epochs"]
    batch_size = cfg["batch_size"]
    hidden = cfg["hidden"]
    heads = cfg["heads"]
    layers = cfg["layers"]
    lr = cfg["lr"]
    weight_decay = cfg["weight_decay"]

    # device selection for training (cpu/gpu)
    device = cfg["device"]
    print(f"Using device: {device}")


    # ---- Load data ----
    train_data, train_targets = load_data(data_path)
    train_loader = DataLoader(
        MetaStatDataset(train_data, train_targets),
        batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    if val_path is not None:
        val_data, val_targets = load_data(val_path)
        val_loader = DataLoader(
            MetaStatDataset(val_data, val_targets),
            batch_size=batch_size, collate_fn=collate_fn
        )
    

    # ---- Model ----
    model = MyTransformerEstimator(
        d_model=hidden, nhead=heads, num_layers=layers
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    base_dir = os.path.join("experiments", name)
    os.makedirs(base_dir, exist_ok=True)
    model_path = os.path.join(base_dir, f"{name}_model.pt")

    best_val = float("inf")
    results = []

    # ---- Training ----
    for epoch in range(1, epochs+1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        log_msg = f"Epoch {epoch:03d} | train loss {train_loss:.6f}"

        if val_path is not None:
            val_loss = eval_epoch(model, val_loader, loss_fn, device)
            
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), model_path)
            log_msg += f" | val loss {val_loss:.6f}"

            results.append({"train_loss": train_loss, "val_loss": val_loss})

        else:
            results.append({"train_loss": train_loss})

        print(log_msg, flush=True)

    # ---- Save model, configuration and results ----
    if val_path is None:
        torch.save(model.state_dict(), model_path)

    config_path = os.path.join(base_dir, f"{name}_config.json")
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=4, default=str)

    results_path = os.path.join(base_dir, f"{name}_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Files saved in: {base_dir}")


if __name__ == "__main__":
    main()
