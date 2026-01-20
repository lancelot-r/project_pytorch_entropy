#!/usr/bin/env python
"""
Usage example
-------------
export PYTORCH_ENABLE_MPS_FALLBACK=1

python code/metadata.py \           
    --config config/config_data.json
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
    "exponential",
    "gamma",
    "lognormal",
    "laplace",
    "logistic",
    "beta",
    "chi_square",
    "cauchy",
    "rayleigh",
    "weibull",
    "pareto",
    "inverse_gamma",
    "student_t"
]


def sample_dataset(allowed_dists: List[str], n=None) -> Tuple[np.ndarray, float, str]:
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

    elif distribution == "laplace":
        mu = np.random.uniform(-3, 3)
        b = np.random.uniform(0.2, 2.0)
        X = np.random.laplace(mu, b, size=n)
        y = 1 + np.log(2 * b)

    elif distribution == "logistic":
        mu = np.random.uniform(-3, 3)
        s = np.random.uniform(0.2, 2.0)
        X = np.random.logistic(mu, s, size=n)
        y = 2 + np.log(s)

    elif distribution == "beta":
        alpha = np.random.uniform(0.5, 5)
        beta_ = np.random.uniform(0.5, 5)
        X = np.random.beta(alpha, beta_, size=n)
        y = (
            np.log(gamma_func(alpha) * gamma_func(beta_) / gamma_func(alpha + beta_))
            - (alpha - 1) * digamma(alpha)
            - (beta_ - 1) * digamma(beta_)
            + (alpha + beta_ - 2) * digamma(alpha + beta_)
        )

    elif distribution == "chi_square":
        k = np.random.uniform(1, 10)
        X = np.random.chisquare(k, size=n)
        y = (
            0.5 * k + np.log(2 * gamma_func(k / 2)) + (1 - k / 2) * digamma(k / 2)
        )

    elif distribution == "cauchy":
        x0 = np.random.uniform(-3, 3)
        gamma_ = np.random.uniform(0.2, 3)
        X = x0 + gamma_ * np.tan(np.pi * (np.random.rand(n) - 0.5))
        y = np.log(4 * np.pi * gamma_)

    elif distribution == "rayleigh":
        sigma = np.random.uniform(0.2, 3)
        X = np.random.rayleigh(sigma, size=n)
        y = 1 + np.log(sigma / np.sqrt(2))

    elif distribution == "weibull":
        k = np.random.uniform(0.5, 5)
        lam = np.random.uniform(0.5, 3)
        X = lam * np.random.weibull(k, size=n)
        y = (
            gamma_func(1 + 1/k)
            - (1 - 1/k) * digamma(1)
            + np.log(lam / k)
            + 1
        )
        # simplified: y = gamma* (1 - 1/k) + log(lam/k) + 1  
        # but gamma* = Euler-Mascheroni constant, use digamma(1) = -gamma*

    elif distribution == "pareto":
        xm = np.random.uniform(0.5, 3)
        alpha = np.random.uniform(1.1, 5)
        X = xm * (1 - np.random.rand(n)) ** (-1 / alpha)
        y = np.log(xm / alpha) + 1 + 1 / alpha

    elif distribution == "inverse_gamma":
        alpha = np.random.uniform(1, 5)
        beta = np.random.uniform(0.5, 3)
        X = 1 / np.random.gamma(alpha, 1 / beta, size=n)
        y = (
            alpha + np.log(beta * gamma_func(alpha))
            - (1 + alpha) * digamma(alpha)
        )

    elif distribution == "student_t":
        nu = np.random.uniform(1.5, 20)
        X = np.random.standard_t(nu, size=n).astype(np.float32)
        y = (
            np.log(np.sqrt(nu) * gamma_func((nu) / 2) / gamma_func((nu + 1) / 2))
            + (nu + 1)/2 * (digamma((nu + 1)/2) - digamma(nu/2))
        )


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

    train_datasets, train_targets, train_dists = generate_meta_dataset(size, distribution, n=n_value)
    train_path = os.path.join(base_dir, f"{name}.npz")
    save_meta_dataset(train_datasets, train_targets, train_dists, train_path)


    # Save the used configuration
    used_cfg_path = os.path.join(base_dir, f"{name}.json")
    with open(used_cfg_path, "w") as f:
        json.dump(cfg, f, indent=4)

    print(f"Files saved in: {base_dir}")

if __name__ == "__main__":
    main()