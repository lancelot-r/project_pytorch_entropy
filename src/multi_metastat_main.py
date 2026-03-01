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
from metastat_dataloader2 import MetaStatDataset, collate_fn, load_data
from multi_metastat_model import MyTransformerEstimator
import numpy as np
import os
import json
from tqdm import tqdm

def train_epoch(model, loader, optimizer, loss_fn, device, accum_steps=1):
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
    parser = argparse.ArgumentParser(description="Meta-Statistical Entropy Estimator Training")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file '{args.config}' not found.")
    
    with open(args.config, "r") as f:
        cfg = json.load(f)
    print(f"Loaded training configuration from {args.config}")

    # Extract parameters from JSON
    base_dir_cfg = cfg.get("dir")

    def with_base_dir(*paths):
        if base_dir_cfg is None:
            return os.path.join(*paths)
        return os.path.join(base_dir_cfg, *paths)
    
    name = cfg["name"]
    data_name = cfg["data"]
    data_path = with_base_dir("data", data_name, f"{data_name}.npz")
    val_name = cfg.get("val_data")
    if val_name is not None:
        val_path = with_base_dir("data", val_name, f"{val_name}.npz")
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
    # Infer feature dimension
    first_sample = train_data[0]
    dim = first_sample.shape[1] if first_sample.ndim == 2 else 1

    if val_path is not None:
        val_data, val_targets = load_data(val_path)
        val_loader = DataLoader(
            MetaStatDataset(val_data, val_targets),
            batch_size=batch_size, collate_fn=collate_fn
        )
    

    # ---- Model ----
    model = MyTransformerEstimator(
        input_dim=dim, d_model=hidden, nhead=heads, num_layers=layers
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    base_dir = with_base_dir("training", name)
    os.makedirs(base_dir, exist_ok=True)
    model_path = os.path.join(base_dir, f"{name}_model.pt")



    # Save configuration
    config_path = os.path.join(base_dir, f"{name}_config.json")
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=4, default=str)

    best_val = float("inf")
    results = []

    patience = cfg.get("early_stopping_patience", 8)
    min_delta = cfg.get("early_stopping_min_delta", 1e-4)

    best_val = float("inf")
    epochs_no_improve = 0


    # ---- Training ----
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        log_msg = f"Epoch {epoch:03d} | train loss {train_loss:.6f}"

        if val_path is not None:
            val_loss = eval_epoch(model, val_loader, loss_fn, device)

            if val_loss < best_val - min_delta:
                best_val = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), model_path)
            else:
                epochs_no_improve += 1

            log_msg += f" | val loss {val_loss:.6f}"
            log_msg += f" | no_improve {epochs_no_improve}/{patience}"

            results.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss
            })

            if epochs_no_improve >= patience:
                print(
                    f"🛑 Early stopping at epoch {epoch} "
                    f"(best val loss {best_val:.6f})"
                )
                break
        else:
            results.append({
                "epoch": epoch,
                "train_loss": train_loss
            })

        print(log_msg, flush=True)


    # ---- Save model and results ----
    if val_path is None:
        torch.save(model.state_dict(), model_path)

    results_path = os.path.join(base_dir, f"{name}_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Files saved in: {base_dir}")


if __name__ == "__main__":
    main()
