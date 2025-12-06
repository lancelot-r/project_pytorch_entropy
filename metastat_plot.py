import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

 
# Chargement des données

df = pd.read_csv("C:\\Users\\utilisateur\\Documents\\stage_transformer\\project_pytorch_entropy\\results\\test_predictions.csv")


# Dossier de sortie pour les figures
output_dir = "plot"
os.makedirs(output_dir, exist_ok=True)


# Vraies entropies
true_entropy = df["entropy"]

# Erreurs du modèle et de scipy par rapport à la vérité
df["model_error"] = df["model_entropy"] - true_entropy
df["scipy_error"] = df["scipy_entropy"] - true_entropy

# Biais = erreur signée (prediction - true)
df["model_bias"] = df["model_error"]
df["scipy_bias"] = df["scipy_error"]

# Erreurs absolues
df["abs_error"] = df["model_error"].abs()  # modèle
df["abs_model_bias"] = df["model_bias"].abs()
df["abs_scipy_bias"] = df["scipy_bias"].abs()

# erreur absolue baseline bootstrap 
df["boot_error"] = df["boot_mean"] - true_entropy
df["abs_boot_error"] = df["boot_error"].abs()

 
# 2. Quelques stats globales (modèle + baseline)

# RMSE et MAE du modèle
rmse_model = np.sqrt((df["model_error"] ** 2).mean())
mae_model = df["abs_error"].mean()

# RMSE et MAE du baseline scipy
rmse_scipy = np.sqrt((df["scipy_error"] ** 2).mean())
mae_scipy = df["abs_scipy_bias"].mean()

# Corrélation entre prédiction du modèle et vérité
corr = df[["model_entropy", "entropy"]].corr().iloc[0, 1]

print(f"Modèle : RMSE global  = {rmse_model:.4f}")
print(f"Modèle : MAE global   = {mae_model:.4f}")
print(f"Scipy  : RMSE global  = {rmse_scipy:.4f}")
print(f"Scipy  : MAE global   = {mae_scipy:.4f}")
print(f"Corrélation modèle / vérité : {corr:.4f}")


 
# 3. Histogramme du biais du modèle
 
plt.figure(figsize=(8, 6))
plt.hist(df["model_bias"], bins=40, edgecolor="black", alpha=0.7)
plt.axvline(0, linestyle="--", label="Biais nul")
plt.xlabel("Model Bias (model_entropy - entropy)", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.title("Distribution du biais du modèle", fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "01_hist_bias.png"), dpi=300)
plt.close()


 
# 4. Prédiction du modèle vs Entropie vraie

x = df["entropy"]          # vérité
y = df["model_entropy"]    # prédiction modèle

min_val = min(x.min(), y.min())
max_val = max(x.max(), y.max())

plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.3, s=15, label="Dataset")
plt.plot([min_val, max_val], [min_val, max_val],
         linestyle="--", label="y = x (prédiction parfaite)")
plt.xlabel("True Entropy (entropy)", fontsize=12)
plt.ylabel("Predicted Entropy (model)", fontsize=12)
plt.title("Prédiction du modèle vs Entropie vraie", fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "02_pred_vs_true.png"), dpi=300)
plt.close()


 
# 5. Erreur absolue du modèle en fonction de la taille n
 
plt.figure(figsize=(8, 6))
plt.scatter(df["n"], df["abs_error"], alpha=0.3, s=15, label="Erreur individuelle")

# On lisse en prenant la moyenne par "bin" de n
num_bins = 15
bins = np.linspace(df["n"].min(), df["n"].max(), num_bins + 1)
bin_indices = np.digitize(df["n"], bins)
bin_centers = []
bin_means = []

for b in range(1, num_bins + 1):
    mask = bin_indices == b
    if mask.sum() > 0:
        bin_centers.append(df["n"][mask].mean())
        bin_means.append(df["abs_error"][mask].mean())

plt.plot(bin_centers, bin_means,
         linewidth=2,
         label="Erreur absolue moyenne (par bin de n)")

plt.xlabel("n (taille du dataset)", fontsize=12)
plt.ylabel("|model_entropy - entropy|", fontsize=12)
plt.title("Erreur absolue du modèle selon la taille du dataset n", fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "03_error_vs_n.png"), dpi=300)
plt.close()


 
# 6. Boxplot SUPERPOSÉ modèle vs baseline par distribution
# On compare visuellement les deux 
 
# On calcule les erreurs absolues par distribution
dist_order = (
    df.groupby("distribution")["abs_error"]
    .median()
    .sort_values()
    .index
)

model_data = [df.loc[df["distribution"] == d, "abs_error"] for d in dist_order]
scipy_data = [df.loc[df["distribution"] == d, "abs_scipy_bias"] for d in dist_order]

plt.figure(figsize=(12, 6))
ax = plt.gca()

indices = np.arange(len(dist_order))

# Décalage horizontal pour séparer les deux boxplots
offset = 0.15

bp1 = ax.boxplot(
    model_data,
    positions=indices - offset,
    widths=0.25,
    showfliers=False
)
bp2 = ax.boxplot(
    scipy_data,
    positions=indices + offset,
    widths=0.25,
    showfliers=False
)

ax.set_xticks(indices)
ax.set_xticklabels(dist_order, rotation=45, ha="right")

plt.ylabel("Absolute Error", fontsize=12)
plt.title("Erreur absolue par distribution : modèle vs baseline scipy", fontsize=14)
plt.grid(True, axis="y", alpha=0.3)
plt.legend([bp1["boxes"][0], bp2["boxes"][0]],
           ["Modèle", "Baseline (scipy)"],
           loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "04_boxplot_abs_error_model_vs_scipy.png"), dpi=300)
plt.close()


 
# 7. Comparaison baseline vs modèle (biais absolu, point par point)
 
plt.figure(figsize=(8, 6))
plt.scatter(df["abs_scipy_bias"], df["abs_model_bias"],
            alpha=0.3, s=15, label="Dataset")

min_b = min(df["abs_scipy_bias"].min(), df["abs_model_bias"].min())
max_b = max(df["abs_scipy_bias"].max(), df["abs_model_bias"].max())

plt.plot([min_b, max_b], [min_b, max_b],
         linestyle="--", label="y = x (même performance)")

plt.xlabel("Baseline absolute bias (scipy)", fontsize=12)
plt.ylabel("Model absolute bias", fontsize=12)
plt.title("Comparaison du biais absolu : modèle vs baseline scipy", fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "05_baseline_vs_model.png"), dpi=300)
plt.close()


 
# 8. Distribution des entropies VRAIES
 
plt.figure(figsize=(8, 6))
plt.hist(df["entropy"], bins=40, edgecolor="black", alpha=0.7)
plt.xlabel("True entropy (entropy)", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.title("Distribution des entropies vraies (test set)", fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "06_true_entropy_distribution.png"), dpi=300)
plt.close()


 
# 9. Erreur absolue du modèle en fonction de l'entropie vraie
 
plt.figure(figsize=(8, 6))
plt.scatter(df["entropy"], df["abs_error"], alpha=0.3, s=15, label="Erreur individuelle")

num_bins_e = 15
bins_e = np.linspace(df["entropy"].min(), df["entropy"].max(), num_bins_e + 1)
bin_indices_e = np.digitize(df["entropy"], bins_e)
bin_centers_e = []
bin_means_e = []

for b in range(1, num_bins_e + 1):
    mask = bin_indices_e == b
    if mask.sum() > 0:
        bin_centers_e.append(df["entropy"][mask].mean())
        bin_means_e.append(df["abs_error"][mask].mean())

plt.plot(bin_centers_e, bin_means_e,
         linewidth=2,
         label="Erreur absolue moyenne (par bin d'entropie vraie)")

plt.xlabel("True Entropy (entropy)", fontsize=12)
plt.ylabel("Absolute Error", fontsize=12)
plt.title("Erreur absolue du modèle en fonction de l'entropie vraie", fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "07_error_vs_true_entropy.png"), dpi=300)
plt.close()


 
# 10. Barplot : erreur absolue moyenne par distribution (modèle)
 
mean_abs_error_by_dist = (
    df.groupby("distribution")["abs_error"]
    .mean()
    .sort_values()
)

plt.figure(figsize=(12, 6))
plt.bar(mean_abs_error_by_dist.index, mean_abs_error_by_dist.values)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Mean Absolute Error (modèle)", fontsize=12)
plt.title("Erreur absolue moyenne par distribution (modèle)", fontsize=14)
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "08_bar_mean_abs_error_by_dist.png"), dpi=300)
plt.close()

print(f"Toutes les figures ont été enregistrées dans le dossier '{output_dir}/'.")
