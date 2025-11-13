# python metadata.py \     
#     --size 1000 \               
#     --outdir data \                
#     --name mini_train.npz \
#     --distribution uniform normal exponential \
#     --n 100 300

import json
import numpy as np
from typing import Tuple, List
import os
import argparse
from scipy.special import gamma as gamma_func, digamma


AVAILABLE_DISTS = ["normal", "uniform", "exponential", "gamma", "lognormal"]


def sample_dataset(allowed_dists: List[str], n=None) -> Tuple[np.ndarray, float, str]:
    """Generate one dataset X and its analytical entropy label from one of the allowed distributions."""
    distribution = np.random.choice(allowed_dists)

    # handle n as scalar or interval
    if n is None:
        n = np.random.randint(10, 301)
    elif isinstance(n, (tuple, list)) and len(n) == 2:
        n = np.random.randint(n[0], n[1] + 1)
    elif isinstance(n, (int, np.integer)):
        pass
    else:
        raise ValueError("n must be an int, a tuple of two ints, or None.")

    # generate dataset and entropy
    if distribution == "normal":
        mu = np.random.uniform(-5, 5)
        sigma = np.random.uniform(0.5, 3)
        X = np.random.normal(mu, sigma, size=n)
        y = 0.5 * np.log(2 * np.pi * np.e * sigma**2)

    elif distribution == "uniform":
        a = np.random.uniform(-5, 0)
        b = np.random.uniform(0, 5)
        if b <= a:
            b = a + 1.0
        X = np.random.uniform(a, b, size=n)
        y = np.log(b - a)

    elif distribution == "exponential":
        lam = np.random.uniform(0.2, 3)
        X = np.random.exponential(1 / lam, size=n)
        y = 1 - np.log(lam)

    elif distribution == "gamma":
        k = np.random.uniform(0.5, 5)
        theta = np.random.uniform(0.5, 3)
        X = np.random.gamma(k, theta, size=n)
        y = k + np.log(theta) + np.log(gamma_func(k)) + (1 - k) * digamma(k)

    elif distribution == "lognormal":
        mu = np.random.uniform(-1, 1)
        sigma = np.random.uniform(0.2, 1.0)
        X = np.random.lognormal(mu, sigma, size=n)
        y = 0.5 + mu + np.log(sigma * np.sqrt(2 * np.pi))

    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    return X.astype(np.float32), np.float32(y), distribution


def generate_meta_dataset(n_samples: int, allowed_dists: List[str], n=None):
    """Generate a full meta-dataset of (dataset, entropy, distribution) triples."""
    datasets, targets, dists = [], [], []
    for _ in range(n_samples):
        X, y, dist = sample_dataset(allowed_dists, n)
        datasets.append(X)
        targets.append(y)
        dists.append(dist)
    return datasets, targets, dists


def save_meta_dataset(datasets, targets, dists, save_path: str):
    """
    Save datasets and targets into a compressed NumPy file (.npz).
    Each dataset may have variable length, so we store them as dtype=object.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(
        save_path,
        datasets=np.array(datasets, dtype=object),
        targets=np.array(targets, dtype=np.float32),
        dists=np.array(dists, dtype=object),
    )
    print(f"Saved meta-dataset to {save_path} ({len(datasets)} samples)")



def summarize_dataset(targets: List[float], dists: List[str]):
    """Print summary statistics for quick inspection."""
    print("\n📊 Summary statistics:")
    print(f"Entropy mean: {np.mean(targets):.4f}, std: {np.std(targets):.4f}")
    unique, counts = np.unique(dists, return_counts=True)
    print("Distribution breakdown:")
    for u, c in zip(unique, counts):
        print(f"  - {u:<12}: {c} samples")
    print()


def main():
    parser = argparse.ArgumentParser(description="Generate Meta-Datasets (JSON-only version)")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
    args = parser.parse_args()

    # --- Load configuration ---
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file '{args.config}' not found.")
    with open(args.config, "r") as f:
        cfg = json.load(f)
    print(f"Loaded configuration from {args.config}")

    # --- Extract parameters ---
    train_size = cfg.get("train_size")
    test_size = cfg.get("test_size")
    test_name = cfg.get("test_name", "test_meta.npz")
    val_size = cfg.get("val_size")
    outdir = cfg.get("outdir", "data")
    train_name = cfg.get("train_name", "train_meta.npz")
    val_name = cfg.get("val_name", "val_meta.npz")
    distribution = cfg.get("distribution", AVAILABLE_DISTS)
    n = cfg.get("n")
    seed = cfg.get("seed")

    if seed is not None:
        np.random.seed(seed)

    # interpret n
    if isinstance(n, list) and len(n) == 2:
        n_value = (n[0], n[1])
    elif isinstance(n, int):
        n_value = n
    else:
        n_value = None

    os.makedirs(outdir, exist_ok=True)

    print(f"Generating training data ({train_size} samples) using distributions: {distribution}")
    train_datasets, train_targets, train_dists = generate_meta_dataset(train_size, distribution, n=n_value)
    train_path = os.path.join(outdir, train_name)
    save_meta_dataset(train_datasets, train_targets, train_dists, train_path)
    summarize_dataset(train_targets, train_dists)

    if val_size:
        print(f"Generating validation data ({val_size} samples) using distributions: {distribution}")
        val_datasets, val_targets, val_dists = generate_meta_dataset(val_size, distribution, n=n_value)
        val_path = os.path.join(outdir, val_name)
        save_meta_dataset(val_datasets, val_targets, val_dists, val_path)
        summarize_dataset(val_targets, val_dists)
        
    if test_size:
        print(f"Generating test data ({test_size} samples) using distributions: {distribution}")
        test_datasets, test_targets, test_dists = generate_meta_dataset(test_size, distribution, n=n_value)
        test_path = os.path.join(outdir, test_name)
        save_meta_dataset(test_datasets, test_targets, test_dists, test_path)
        summarize_dataset(test_targets, test_dists)

    # Save the used configuration
    used_cfg_path = os.path.join(outdir, "used_config.json")
    with open(used_cfg_path, "w") as f:
        json.dump(cfg, f, indent=4)
    print(f"Saved configuration to {used_cfg_path}")
    print(f"Files saved in: {os.path.abspath(outdir)}")

if __name__ == "__main__":
    main()
