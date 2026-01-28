import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_var_vs_n(
    n,
    var,
    title,
    ylabel,
    output_dir,
    filename,
    size=(30, 300),
):
    figsize=(8, 6)
    alpha_scatter=0.3
    scatter_size=15
    num_bins=15

    n = np.asarray(n)
    var = np.asarray(var)

    # --------------------
    # Scatter plot
    # --------------------
    plt.figure(figsize=figsize)
    plt.scatter(
        n,
        var,
        alpha=alpha_scatter,
        s=scatter_size,
        label="Erreur individuelle",
    )

    # --------------------
    # Binning over n
    # --------------------
    bins = np.linspace(n.min(), n.max(), num_bins + 1)
    bin_indices = np.digitize(n, bins)

    bin_centers = []
    bin_means = []

    for b in range(1, num_bins + 1):
        mask = bin_indices == b
        if mask.sum() > 0:
            bin_centers.append(n[mask].mean())
            bin_means.append(var[mask].mean())

    plt.plot(
        bin_centers,
        bin_means,
        linewidth=2,
        label="MSE (par bin de n)",
        color="black"
    )

    # --------------------
    # Vertical reference lines
    # --------------------
    n_min, n_max = size
    plt.axvline(
        n_min,
        color="red",
        linestyle=":",
        linewidth=2,
        label=f"n = {n_min}",
    )
    plt.axvline(
        n_max,
        color="red",
        linestyle=":",
        linewidth=2,
        label=f"n = {n_max}",
    )

    # --------------------
    # Formatting
    # --------------------
    plt.xlabel("n (taille du dataset)", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(
            os.path.join(output_dir, filename),
            dpi=300,
        )



def plot_boxplot_by_distribution(
    df,
    var_model,
    var_knn,
    title,
    ylabel,
    output_dir,
    filename
):

    figsize=(14, 6)

    var_model = np.asarray(var_model)
    var_knn = np.asarray(var_knn)

    # --------------------
    # Build working DataFrame
    # --------------------
    work_df = df.copy()
    work_df["var_model"] = var_model
    work_df["var_knn"] = var_knn

    # --------------------
    # Order distributions by median model error
    # --------------------
    dist_order = (
        work_df.groupby("distribution")["var_model"]
        .median()
        .sort_values()
        .index
        .tolist()
    )

    # Add global distribution
    dist_order.append("ALL")

    # --------------------
    # Collect data
    # --------------------
    model_data = []
    knn_data = []

    for d in dist_order:
        if d == "ALL":
            model_data.append(work_df["var_model"])
            knn_data.append(work_df["var_knn"])
        else:
            model_data.append(
                work_df.loc[work_df["distribution"] == d, "var_model"]
            )
            knn_data.append(
                work_df.loc[work_df["distribution"] == d, "var_knn"]
            )

    # --------------------
    # Plot
    # --------------------
    fig, ax = plt.subplots(figsize=figsize)

    indices = np.arange(len(dist_order))
    offset = 0.18

    bp_model = ax.boxplot(
        model_data,
        positions=indices - offset,
        widths=0.3,
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor="red", alpha=0.6),
        medianprops=dict(color="black"),
    )

    bp_knn = ax.boxplot(
        knn_data,
        positions=indices + offset,
        widths=0.3,
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor="black", alpha=0.6),
        medianprops=dict(color="white"),
    )

    # --------------------
    # Axes & labels
    # --------------------
    ax.set_xticks(indices)
    ax.set_xticklabels(dist_order, rotation=45, ha="right")

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, axis="y", alpha=0.3)

    # Legend
    ax.legend(
        [bp_model["boxes"][0], bp_knn["boxes"][0]],
        ["Transformer Model", "knn Baseline"],
        loc="upper right",
    )

    plt.tight_layout()

    # --------------------
    # Save
    # --------------------
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(
            os.path.join(output_dir, filename),
            dpi=300,
        )



def main():
    parser = argparse.ArgumentParser(description="Meta-Statistical Entropy Estimator Visualisation")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
    args = parser.parse_args()


    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file '{args.config}' not found.")
    
    with open(args.config, "r") as f:
        cfg = json.load(f)
    print(f"Loaded training configuration from {args.config}")


    results_name = cfg["results"]
    results_path = os.path.join("evaluation", results_name, f"{results_name}.csv")
    save_name = cfg["save"]
    base_dir = os.path.join("visualisations", save_name)
    os.makedirs(base_dir, exist_ok=True)


    df = pd.read_csv(results_path)


    # MSE
    model_mse = np.mean((df["model_entropy"] - df["entropy"]) ** 2)
    knn_mse = np.mean((df["knn_entropy"] - df["entropy"]) ** 2)
    print("Point-estimate MSE:")
    print(f"  Model  MSE = {model_mse:.6f}")
    print(f"  knn  MSE = {knn_mse:.6f}")


    # Remove rows with -inf (+inf and NaN) for knn_entropy_boot
    n_initial = len(df)
    mask = np.isfinite(df["knn_entropy_boot"])
    df_clean = df[mask]
    n_final = len(df_clean)
    n_removed = n_initial - n_final
    print(
        f"Removed {n_removed} rows out of {n_initial} "
        f"({100 * n_removed / n_initial:.2f}%) due to non-finite knn_entropy_boot."
    )


    # Model MSE vs n
    plot_var_vs_n(
        n=df["n"],
        var=(df["model_entropy"] - df["entropy"])**2,
        title="MSE du modèle selon n",
        ylabel="MSE",
        output_dir=base_dir,
        filename=f"{save_name}_MSEmodel_vs_n.png"
    )

    # knn MSE vs n
    plot_var_vs_n(
        n=df["n"],
        var=(df["knn_entropy"] - df["entropy"])**2,
        title="MSE du knn selon n",
        ylabel="MSE",
        output_dir=base_dir,
        filename=f"{save_name}_MSEknn_vs_n.png",
    )

    # Boxplot MSE
    plot_boxplot_by_distribution(
        df=df,
        var_model=(df["model_entropy"] - df["entropy"])**2,
        var_knn=(df["knn_entropy"] - df["entropy"])**2,
        title="MSE par distribution: Modèle vs knn",
        ylabel="MSE",
        output_dir=base_dir,
        filename=f"{save_name}_boxplot_MSE.png"
    )

    # Boxplot biais2
    plot_boxplot_by_distribution(
        df=df_clean,
        var_model= df_clean["model_bias2"],
        var_knn=df_clean["knn_bias2"],
        title="Biais2 par distribution : Modèle vs knn",
        ylabel="Biais2",
        output_dir=base_dir,
        filename=f"{save_name}_boxplot_biais2.png"
    )

    # Boxplot variance
    plot_boxplot_by_distribution(
        df=df_clean,
        var_model= df_clean["model_var"],
        var_knn=df_clean["knn_var"],
        title="Variance par distribution : Modèle vs knn",
        ylabel="Variance",
        output_dir=base_dir,
        filename=f"{save_name}_boxplot_var.png"
    )

    plt.show()



if __name__ == "__main__":
    main()