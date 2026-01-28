#!/usr/bin/env python
"""
Usage example
-------------
export PYTORCH_ENABLE_MPS_FALLBACK=1

python code/metadata2.py \           
    --config config/config_data2.json
"""

import json
import numpy as np
from typing import Tuple, List
import os
import argparse
from scipy.special import gamma as gamma_func, digamma




AVAILABLE_DISTS = [
    "normal",
    "uniform",
    "student_t",
    "dirichlet"
]




def sample_dataset(allowed_dists: List[str], n=None, dim:int=1) -> Tuple[np.ndarray, float, str]:
    """Generate one dataset X and its analytical entropy label from one of the allowed distributions."""
    
    distribution = np.random.choice(allowed_dists)


    # handle n as scalar or interval
    if n is None:
        n = np.random.randint(10, 301)
    elif isinstance(n, (tuple, list)) and len(n) == 2:
        n_min, n_max = n
        # log-uniform sampling for better variety 
        n = int(np.exp(np.random.uniform(np.log(n_min), np.log(n_max))))
    elif isinstance(n, (int, np.integer)):
        pass
    else:
        raise ValueError("n must be an int, a tuple of two ints, or None.")


    # generate dataset and entropy
    if distribution == "normal":
        mu = np.random.uniform(-5, 5, size=dim)
        variances = np.random.uniform(0.5, 3, size=dim)
        Sigma = np.diag(variances)
        X = np.random.multivariate_normal(mu, Sigma, size=n)
        # y = 0.5 * np.log((2 * np.pi * np.e)**dim * np.linalg.det(Sigma))
        y = 0.5 * (dim * np.log(2 * np.pi * np.e) + np.sum(np.log(variances)))    # more stable in high-dimension


    elif distribution == "uniform":
        a = np.random.uniform(-5, 0, size=dim)
        b = np.random.uniform(0, 5, size=dim)
        b = np.maximum(b, a + 1.0)
        X = np.random.uniform(a, b, size=(n, dim))
        y = np.sum(np.log(b - a))


    elif distribution == "student_t":
        nu = np.random.uniform(3.0, 20.0)          # degrees of freedom
        mu = np.random.uniform(-5, 5, size=dim)

        variances = np.random.uniform(0.5, 3, size=dim)
        Sigma = np.diag(variances)

        # Sampling
        Z = np.random.multivariate_normal(np.zeros(dim), Sigma, size=n)
        U = np.random.chisquare(nu, size=n)
        X = mu + Z / np.sqrt(U[:, None] / nu)

        # Entropy
        y = (
            np.log(gamma_func((nu + dim) / 2))
            - np.log(gamma_func(nu / 2))
            + 0.5 * (
                dim * np.log(nu * np.pi)
                + np.sum(np.log(variances))
            )
            + (nu + dim) / 2 * (
                digamma((nu + dim) / 2)
                - digamma(nu / 2)
            )
        )


    elif distribution == "dirichlet":
        alpha = np.random.uniform(0.5, 5.0, size=dim)
        alpha0 = np.sum(alpha)

        X = np.random.dirichlet(alpha, size=n)

        y = (
            np.sum(np.log(gamma_func(alpha)))
            - np.log(gamma_func(alpha0))
            + (alpha0 - dim) * digamma(alpha0)
            - np.sum((alpha - 1) * digamma(alpha))
        )


    else:
        raise ValueError(f"Unknown distribution: {distribution}")


    return X.astype(np.float32), np.float32(y), distribution





def generate_meta_dataset(n_samples: int, allowed_dists: List[str], n=None, dim=1):
    """Generate a full meta-dataset of (dataset, entropy, distribution) triples."""
    datasets, targets, dists = [], [], []
    for _ in range(n_samples):
        X, y, dist = sample_dataset(allowed_dists, n, dim=dim)
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
    size = cfg.get("size")
    name = cfg.get("name", "metadata.npz")
    distribution = cfg.get("distribution", AVAILABLE_DISTS)
    n = cfg.get("n")
    dim = cfg.get("dim", 1)
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

    base_dir = os.path.join("data", name)
    os.makedirs(base_dir, exist_ok=True)

    train_datasets, train_targets, train_dists = generate_meta_dataset(size, distribution, n=n_value,dim=dim)
    train_path = os.path.join(base_dir, f"{name}.npz")
    save_meta_dataset(train_datasets, train_targets, train_dists, train_path)


    # Save the used configuration
    used_cfg_path = os.path.join(base_dir, f"{name}.json")
    with open(used_cfg_path, "w") as f:
        json.dump(cfg, f, indent=4)

    print(f"Files saved in: {base_dir}")

if __name__ == "__main__":
    main()