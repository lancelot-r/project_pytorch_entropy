# Pipeline univariée

## Génération de méta-jeux de données

Nous avons commencé par écrire un script (`code/metadata.py`) qui permet de générer un méta-jeu de données.  
Un fichier JSON (`config/config_data.json`) permet de choisir les paramètres de génération du méta-jeu de données.

### Paramètres de `config_data.json`

| Paramètre | Type | Description |
|------------|------|------------|
| `size` | int | nombre total d'exemples générés dans le méta-jeu de données |
| `name` | str | identifiant utilisé pour l'enregistrement du méta-jeu de données |
| `distributions` | List[str] | liste des distributions utilisées pour générer les échantillons |
| `n` | int ou `[n_min, n_max]` | taille fixe des échantillons ou intervalle de tailles possibles |
| `seed` | int | graine aléatoire assurant la reproductibilité |

Distributions disponibles :

```
["normal", "uniform", "exponential", "gamma", "lognormal",
 "laplace", "logistic", "beta", "chi_square", "cauchy",
 "rayleigh", "weibull", "pareto", "inverse_gamma", "student_t"]
```

Pour générer un méta-jeu de données :

```bash
python code/metadata.py --config config/config_data.json
```

La génération crée un dossier contenant :

- la configuration utilisée
- le méta-jeu de données au format `.npz`

---

## Entraînement du modèle

Le script `code/metastat_main.py` sert à l'entraînement d'un modèle Transformer à partir de méta-jeux de données d'entraînement et de validation.

Il utilise :

- `code/metastat_dataloader.py`
- `code/metastat_model.py`

Chaque échantillon est traité comme une séquence d’observations.  
Comme les longueurs varient, un padding est appliqué pour aligner les séquences dans un même batch.

Le modèle est un petit Transformer sans encodage positionnel qui agrège les représentations pour produire une prédiction scalaire.

Le fichier `config/config_train.json` permet de choisir les paramètres d'entraînement.

### Paramètres de `config_train.json`

| Paramètre | Type | Description |
|------------|------|------------|
| `name` | str | identifiant du modèle entraîné |
| `data` | str | dossier du méta-jeu de données d'entraînement |
| `val_data` | str | dossier du méta-jeu de données de validation |
| `epochs` | int | nombre total d'epochs |
| `batch_size` | int | taille des mini-batchs |
| `hidden` | int | dimension interne du Transformer (`d_model`) |
| `heads` | int | nombre de têtes d’attention |
| `layers` | int | nombre de couches Transformer |
| `lr` | float | taux d'apprentissage |
| `weight_decay` | float | régularisation L2 |
| `device` | str | périphérique (`cpu`, `cuda`, `mps`, etc.) |

Pour lancer l'entraînement :

```bash
python code/metastat_main.py --config config/config_train.json
```

À la fin de l'entraînement, un dossier est créé dans `training/` contenant :

- la configuration utilisée
- le meilleur modèle (`.pt`)
- les résultats MSE entraînement/validation au format JSON

---

## Évaluation

Le script `code/evaluation.py` permet d'évaluer un modèle et de le comparer à la baseline.

Il calcule :

- le MSE
- le biais empirique (bootstrap)
- la variance empirique (bootstrap)

Le fichier `config/config_eval.json` permet de choisir les paramètres d'évaluation.

### Paramètres de `config_eval.json`

| Paramètre | Type | Description |
|------------|------|------------|
| `name` | str | identifiant des résultats |
| `data` | str | dossier du méta-jeu de données d'évaluation |
| `model` | str | dossier contenant les poids du modèle |
| `batch_size` | int | taille des mini-batchs |
| `device` | str | périphérique (`cpu`, `cuda`, `mps`, etc.) |
| `bootstrap` | int | nombre de rééchantillonnages bootstrap |

Pour lancer l'évaluation :

```bash
python code/evaluation.py --config config/config_eval.json
```

Un dossier est créé dans `evaluation/` contenant :

- la configuration d'évaluation
- les résultats MSE, biais et variance au format `.csv`

---

# Pipeline multivariée

La pipeline multivariée est construite de manière similaire.

## Génération de méta-jeux multivariés

Le script `code/metadata2.py` permet de générer des méta-jeux de données multivariés.

Un paramètre supplémentaire est requis :

| Paramètre | Type | Description |
|------------|------|------------|
| `dim` | int | dimension des échantillons multivariés |

Distributions disponibles :

```
["normal", "uniform", "student_t", "dirichlet"]
```

---

## Entraînement multivarié

Script :

```bash
python code/metastat_main2.py --config config/config_train.json
```

Modules utilisés :

- `code/metastat_dataloader2.py`
- `code/metastat_model2.py`

---

## Évaluation multivariée

Script :

```bash
python code/evaluation2.py --config config/config_eval.json
```

Paramètre supplémentaire :

| Paramètre | Type | Description |
|------------|------|------------|
| `knn_k` | int | nombre de voisins pour l'estimation KNN (baseline NPEET) |