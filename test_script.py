import numpy as np
import torch
import csv
from scipy.stats import entropy
from scipy.stats import gaussian_kde

def estimate_entropy_scipy(x):
    kde = gaussian_kde(x)
    grid = np.linspace(np.min(x), np.max(x), 512)
    pdf = kde(grid)
    pdf = pdf / np.trapz(pdf, grid)
    return -np.trapz(pdf * np.log(pdf + 1e-12), grid)

class EntropyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.load("models/metastat_transformer.pt")
    def forward(self, x):
        return self.net(x)

def load_test_data(path):
    data = np.load(path, allow_pickle=True)
    return data["datasets"], data["targets"], data["dists"]

def prepare_input(x):
    x = torch.tensor(x, dtype=torch.float32)
    x = x.unsqueeze(0)
    return x

def main():
    test_datasets, true_entropy, dists = load_test_data("data/test_meta.npz")
    model = EntropyModel()
    model.eval()

    rows = []
    for i, x in enumerate(test_datasets):
        x_input = prepare_input(x)
        with torch.no_grad():
            pred = model(x_input).item()
        ref = estimate_entropy_scipy(x)
        rows.append([i, pred, ref, true_entropy[i], dists[i]])

    with open("test_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "model_entropy", "scipy_entropy", "true_entropy", "distribution"])
        writer.writerows(rows)

if __name__ == "__main__":
    main()