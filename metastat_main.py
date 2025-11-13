# export PYTORCH_ENABLE_MPS_FALLBACK=1

#  python main.py \
#     --data data/train_meta.npz \
#     --val-data data/test_meta.npz \
#     --epochs 100 \
#     --batch-size 8 \
#     --hidden 128 \
#     --layers 4 \
#     --lr 1e-3 \
#     --weight-decay 1e-5 \
#     --eval



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



def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    for x, lengths, y in loader:
        x, lengths, y = x.to(device), lengths.to(device), y.to(device)
        pred = model(x, lengths)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.inference_mode()
def evaluate_mse(model, loader, device):
    model.eval()
    mse = 0
    for x, lengths, y in loader:
        x, lengths, y = x.to(device), lengths.to(device), y.to(device)
        pred = model(x, lengths)
        mse += nn.functional.mse_loss(pred, y).item()
    return mse / len(loader)


@torch.inference_mode()
def evaluate_bias_variance(model, loader, device, K=30):
    model.eval()
    biases, variances = [], []

    for x, lengths, y in loader:
        x, lengths, y = x.to(device), lengths.to(device), y.to(device)

        y_hat_k = []
        for _ in range(K):
            # bootstrap each sequence while keeping shape consistent
            boot_samples = []
            for i in range(len(x)):
                L = lengths[i].item()
                idx = np.random.choice(L, L, replace=True)
                boot = x[i, :L][idx]  # resample within original length
                # pad back to full sequence length for stacking
                pad_len = x.shape[1] - L
                if pad_len > 0:
                    boot = torch.cat([boot, torch.zeros(pad_len, x.shape[2], device=device)], dim=0)
                boot_samples.append(boot)
            x_boot = torch.stack(boot_samples)

            y_hat_k.append(model(x_boot, lengths))

        y_hat_k = torch.stack(y_hat_k)
        y_mean = y_hat_k.mean(0)
        bias2 = ((y_mean - y) ** 2).mean().item()
        var = ((y_hat_k - y_mean) ** 2).mean().item()
        biases.append(bias2)
        variances.append(var)

    return np.mean(biases), np.mean(variances)



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
    data_path = cfg["data"]
    val_path = cfg["val_data"]
    epochs = cfg["epochs"]
    batch_size = cfg["batch_size"]
    hidden = cfg["hidden"]
    heads = cfg["heads"]
    layers = cfg["layers"]
    lr = cfg["lr"]
    weight_decay = cfg["weight_decay"]
    eval_mode = cfg.get("eval", False)
    save_model = cfg.get("save_model")
    save_results = cfg.get("save_results")

    # Select device
    # ---- Device selection (with MPS fallback fix) ----
    # if torch.cuda.is_available():
    #     device = "cuda"
    # elif torch.backends.mps.is_available():
    #     os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    #     device = "mps"
    #     print("Using MPS with CPU fallback for unsupported operations.")
    # else:
    device = cfg["device"]

    print(f"Using device: {device}")


    # ---- Load data ----
    train_data, train_targets = load_data(data_path)
    val_data, val_targets = load_data(val_path)

    train_loader = DataLoader(
        MetaStatDataset(train_data, train_targets),
        batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
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

    # ---- Training ----
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batch_count = 0

        for x, lengths, y in train_loader:
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            pred = model(x, lengths)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1

        avg_train_loss = total_loss / batch_count
        val_mse = evaluate_mse(model, val_loader, device)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val MSE: {val_mse:.4f}")

    # ---- Save model ----
    if save_model:
        os.makedirs(os.path.dirname(save_model), exist_ok=True)
        torch.save(model.state_dict(), save_model)
        print(f"Model weights saved to: {save_model}")

    # ---- Evaluate ----
    results = {"epochs": epochs, "val_mse": evaluate_mse(model, val_loader, device)}
    if eval_mode:
        bias2, var = evaluate_bias_variance(model, val_loader, device)
        results.update({"bias2": bias2, "variance": var})
        print(f"\nBias²: {bias2:.6f} | Variance: {var:.6f}")

    # ---- Save results ----
    if save_results:
        os.makedirs(os.path.dirname(save_results), exist_ok=True)
        with open(save_results, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {save_results}")

    # ---- Save used configuration ----
    config_copy = os.path.join(os.path.dirname(save_results or save_model or "."), "used_train_config.json")
    with open(config_copy, "w") as f:
        json.dump(cfg, f, indent=4)
    print(f"Saved configuration copy to: {config_copy}")


if __name__ == "__main__":
    main()
