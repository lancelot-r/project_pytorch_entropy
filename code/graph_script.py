import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. LOAD CSV
# ============================================================
df = pd.read_csv("results/test_predictions.csv")

# Vérification rapide
print(df.head())

# ============================================================
# 2. GRAPH: ENTROPY vs SAMPLE SIZE
# ============================================================
plt.figure(figsize=(8, 6))
sns.lineplot(data=df, x="n", y="model_entropy", label="Model entropy")
sns.lineplot(data=df, x="n", y="scipy_entropy", label="Scipy entropy")

plt.title("Entropie estimée en fonction de la taille d'échantillon")
plt.xlabel("Taille d'échantillon (n)")
plt.ylabel("Entropie")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# 3. GRAPH: ENTROPY vs DISTRIBUTION TYPE
# ============================================================

# Moyenne par distribution
df_grouped = df.groupby("distribution")[["model_entropy", "scipy_entropy"]].mean().reset_index()

plt.figure(figsize=(10, 6))
df_grouped_melted = df_grouped.melt(id_vars="distribution",
                                    value_vars=["model_entropy", "scipy_entropy"],
                                    var_name="method",
                                    value_name="entropy")

sns.barplot(data=df_grouped_melted, x="distribution", y="entropy", hue="method")

plt.title("Entropie moyenne par type de distribution")
plt.xlabel("Distribution")
plt.ylabel("Entropie moyenne")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ============================================================
# 4. OPTIONAL: scatter plot comparing both methods
# ============================================================
plt.figure(figsize=(7, 7))
sns.scatterplot(data=df, x="scipy_entropy", y="model_entropy", hue="distribution")

plt.title("Comparaison entre l'entropie du modèle et celle de SciPy")
plt.xlabel("Entropie SciPy")
plt.ylabel("Entropie modèle")
plt.grid(True)
plt.tight_layout()
plt.show()
