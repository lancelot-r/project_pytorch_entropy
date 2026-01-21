#!/usr/bin/env python
"""
Usage example
-------------
export PYTORCH_ENABLE_MPS_FALLBACK=1

python code/evaluation.py \
    --config config/config_eval.json
"""


import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from scipy.stats import differential_entropy
from metastat_dataloader import MetaStatDataset, collate_fn, load_data
from metastat_model import MyTransformerEstimator
import torch.nn as nn
import os
import json



@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    preds = []
    lengths = []
    ys = []
    for x, length, y in loader:
        x, length = x.to(device), length.to(device)
        y_hat = model(x, length)
        preds.extend(y_hat.cpu().numpy().tolist())
        lengths.extend(length.cpu().numpy().tolist())
        ys.extend(y.cpu().numpy().tolist())
    return np.array(preds), np.array(lengths), np.array(ys)


def compute_scipy_entropy(sequences):
    return np.array([differential_entropy(seq, method="auto") for seq in sequences])    # other methods?


def bootstrap_indices(n, k):
    return np.random.randint(0, n, size=(k, n))


@torch.inference_mode()
def predict_bootstrap(model, loader, device, k=30):
    model.eval()

    # --- Point predictions (no bootstrap) ---
    model_point_preds = []
    scipy_point_preds = []

    # --- Bootstrap summaries ---
    model_means = []
    scipy_means = []
    model_bias2 = []
    model_var = []
    scipy_bias2 = []
    scipy_var = []

    lengths = []
    true_entropy = []

    for x, length, y in loader:
        x = x.to(device)
        length = length.to(device)

        x_np = x.cpu().numpy()
        length_np = length.cpu().numpy()
        y_np = y.cpu().numpy()

        lengths.extend(length_np.tolist())
        true_entropy.extend(y_np.tolist())

        for xi, li, y_true in zip(x_np, length_np, y_np):
            seq = xi[:li]

            # ----------------------------
            # Point predictions
            # ----------------------------
            if li < 2:
                model_point_preds.append(np.nan)
                scipy_point_preds.append(np.nan)

                model_means.append(np.nan)
                scipy_means.append(np.nan)
                model_bias2.append(np.nan)
                model_var.append(np.nan)
                scipy_bias2.append(np.nan)
                scipy_var.append(np.nan)
                continue

            # Model (point)
            xb = torch.tensor(
                seq, dtype=torch.float32, device=device
            ).unsqueeze(0)
            lb = torch.tensor([li], device=device)
            model_point = model(xb, lb).item()
            model_point_preds.append(model_point)

            # SciPy (point)
            scipy_point = differential_entropy(seq, method="auto")
            scipy_point_preds.append(scipy_point)

            # ----------------------------
            # Bootstrap
            # ----------------------------
            idx = np.random.randint(0, li, size=(k, li))

            model_preds = []
            scipy_preds = []

            for b_idx in idx:
                boot = seq[b_idx]

                xb = torch.tensor(
                    boot, dtype=torch.float32, device=device
                ).unsqueeze(0)
                lb = torch.tensor([len(boot)], device=device)
                model_preds.append(model(xb, lb).item())

                scipy_preds.append(
                    differential_entropy(boot, method="auto")
                )

            model_preds = np.array(model_preds)
            scipy_preds = np.array(scipy_preds)

            model_mean = model_preds.mean()
            scipy_mean = scipy_preds.mean()

            model_means.append(model_mean)
            scipy_means.append(scipy_mean)

            model_bias2.append((model_mean - y_true) ** 2)
            scipy_bias2.append((scipy_mean - y_true) ** 2)

            model_var.append(np.mean((model_preds - model_mean) ** 2))
            scipy_var.append(np.mean((scipy_preds - scipy_mean) ** 2))

    return (
        np.array(model_point_preds),
        np.array(scipy_point_preds),
        np.array(model_means),
        np.array(scipy_means),
        np.array(model_bias2),
        np.array(scipy_bias2),
        np.array(model_var),
        np.array(scipy_var),
        np.array(lengths),
        np.array(true_entropy),
    )




def main():
    parser = argparse.ArgumentParser(description="Meta-Statistical Entropy Estimator Testing")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
    args = parser.parse_args()

    # --- Load configuration ---
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file '{args.config}' not found.")
    with open(args.config, "r") as f:
        cfg = json.load(f)
    print(f"Loaded configuration from {args.config}")

    # --- Extract parameters ---
    name = cfg["name"]
    data_name = cfg["data"]
    data_path = os.path.join("data", data_name, f"{data_name}.npz")
    model_name = cfg["model"]
    model_path = os.path.join("training", model_name, f"{model_name}_model.pt")
    batch_size = cfg["batch_size"]
    device = torch.device(cfg["device"])
    bootstrap_K = cfg.get("bootstrap", None)

    
    npz = np.load(data_path, allow_pickle=True)
    dists = npz["dists"]

    test_data, test_targets = load_data(data_path)
    test_loader = DataLoader(
        MetaStatDataset(test_data, test_targets),
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False   # some order as in scipy_preds?
    )


    model_config = os.path.join("training", model_name, f"{model_name}_config.json")
    with open(model_config, "r") as f:
        model_cfg = json.load(f)
    model = MyTransformerEstimator(
        d_model=model_cfg["hidden"], nhead=model_cfg["heads"], num_layers=model_cfg["layers"]
    )
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)


    if bootstrap_K is not None:
        (
            model_preds,
            scipy_preds,
            model_mean,
            scipy_mean,
            model_bias2,
            scipy_bias2,
            model_var,
            scipy_var,
            lengths,
            true_entropy,
        ) = predict_bootstrap(model, test_loader, device)

    else:
        model_preds, scipy_preds, lengths, true_entropy = predict(model, test_loader, device)
        scipy_preds = compute_scipy_entropy(test_data)


    df = pd.DataFrame({
        "dataset_id": np.arange(len(model_preds)),
        "n": np.asarray(lengths).ravel(),
        "distribution": dists,
        "entropy": np.asarray(true_entropy).ravel(),
        "model_entropy": np.asarray(model_preds).ravel(),
        "scipy_entropy": np.asarray(scipy_preds).ravel(),
    })

    if bootstrap_K is not None:
        df["model_entropy_boot"] = np.asarray(model_mean).ravel()
        df["scipy_entropy_boot"] = np.asarray(scipy_mean).ravel()
        df["model_bias2"] = np.asarray(model_bias2).ravel()
        df["scipy_bias2"] = np.asarray(scipy_bias2).ravel()
        df["model_var"] = np.asarray(model_var).ravel()
        df["scipy_var"] = np.asarray(scipy_var).ravel()


    # guarantee that dists aligns with samples
    assert len(dists) == len(model_preds)
    if bootstrap_K is not None:
        assert len(model_mean) == len(model_preds)


    base_dir = os.path.join("evaluation", name)
    os.makedirs(base_dir, exist_ok=True)
    save_path = os.path.join(base_dir, f"{name}.csv")
    df.to_csv(save_path, index=False)

    config_path = os.path.join(base_dir, f"{name}_config.json")
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=4, default=str)

    print(f"Files saved in: {base_dir}")




if __name__ == "__main__":
    main()