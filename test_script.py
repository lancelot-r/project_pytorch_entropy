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

@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    preds = []
    for x, lengths, _ in loader:
        x, lengths = x.to(device), lengths.to(device)
        y_hat = model(x, lengths)
        preds.extend(y_hat.cpu().numpy().tolist())
    return np.array(preds)

def compute_scipy_entropy(sequences):
    return np.array([differential_entropy(seq) for seq in sequences])

def main():
    parser = argparse.ArgumentParser(description="Meta-Statistical Entropy Estimator Testing")
    parser.add_argument("--test_data", type=str, default="data/test_meta.npz")
    parser.add_argument("--model_file", type=str, default="models/metastat_transformer.pt")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_csv", type=str, default="results/test_predictions.csv")
    args = parser.parse_args()
    
    npz = np.load(args.test_data, allow_pickle=True)
    dists = npz["dists"]

    test_data, test_targets = load_data(args.test_data)
    test_loader = DataLoader(
        MetaStatDataset(test_data, test_targets),
        batch_size=args.batch_size,
        collate_fn=collate_fn
    )

    model = torch.load(args.model_file, map_location=args.device, weights_only = False)
    model.to(args.device)

    model_preds = predict(model, test_loader, args.device)

    sequences = test_data
    lengths = [len(seq) for seq in test_data]
    scipy_est = compute_scipy_entropy(sequences)

    df = pd.DataFrame({
    "dataset_id": np.arange(len(model_preds)),
    "n": lengths,
    "distribution": dists,
    "model_entropy": model_preds.flatten(),
    "scipy_entropy": scipy_est
    })
    
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved predictions to {args.output_csv}")


if __name__ == "__main__":
    main()