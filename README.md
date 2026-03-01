# Available Make Commands

- `make` — Run the default univariate pipeline (data → train → eval)
- `make all` — Same as `make`
- `make univariate` — Run the full univariate pipeline
- `make multivariate` — Run the full multivariate pipeline

## Univariate Steps

- `make data` — Generate univariate metadata
- `make train` — Train univariate model
- `make eval` — Evaluate univariate model

## Multivariate Steps

- `make data_multi` — Generate multivariate metadata
- `make train_multi` — Train multivariate model
- `make eval_multi` — Evaluate multivariate model

## Maintenance

- `make clean` — Remove all generated training and evaluation outputs



# Univariate Pipeline

## Meta-Dataset Generation

We first wrote a script (`src/metadata.py`) to generate a meta-dataset.  
A JSON file (`config/config_data.json`) allows you to specify the parameters for generating the meta-dataset.

### Parameters of `config_data.json`

| Parameter | Type | Description |
|------------|------|------------|
| `size` | int | Total number of examples generated in the meta-dataset |
| `name` | str | Identifier used when saving the meta-dataset |
| `distributions` | List[str] | List of distributions used to generate the samples |
| `n` | int or `[n_min, n_max]` | Fixed sample size or range of possible sample sizes |
| `seed` | int | Random seed ensuring reproducibility |

Available distributions:

```python
["normal", "uniform", "exponential", "gamma", "lognormal",
 "laplace", "logistic", "beta", "chi_square", "cauchy",
 "rayleigh", "weibull", "pareto", "inverse_gamma", "student_t"]
```

The generation process creates a folder containing:

- The configuration used  
- The meta-dataset in `.npz` format  

---

## Model Training

The script `src/metastat_main.py` is used to train a Transformer model from training and validation meta-datasets.

It uses:

- `src/metastat_dataloader.py`  
- `src/metastat_model.py`  

Each sample is treated as a sequence of observations.  
Since sequence lengths vary, padding is applied to align sequences within a batch.

The model is a small Transformer without positional encoding that aggregates representations to produce a scalar prediction.

The file `config/config_train.json` allows you to specify the training parameters.

### Parameters of `config_train.json`

| Parameter | Type | Description |
|------------|------|------------|
| `name` | str | Identifier of the trained model |
| `data` | str | Folder containing the training meta-dataset |
| `val_data` | str | Folder containing the validation meta-dataset |
| `epochs` | int | Total number of epochs |
| `batch_size` | int | Mini-batch size |
| `hidden` | int | Internal Transformer dimension (`d_model`) |
| `heads` | int | Number of attention heads |
| `layers` | int | Number of Transformer layers |
| `lr` | float | Learning rate |
| `weight_decay` | float | L2 regularization |
| `device` | str | Device (`cpu`, `cuda`, `mps`, etc.) |

At the end of training, a folder is created in `training/` containing:

- The configuration used  
- The best model (`.pt`)  
- Training/validation MSE results in JSON format  

---

## Evaluation

The script `src/evaluation.py` evaluates a model and compares it to the baseline.

It computes:

- MSE  
- Empirical bias (bootstrap)  
- Empirical variance (bootstrap)  

The file `config/config_eval.json` allows you to specify evaluation parameters.

### Parameters of `config_eval.json`

| Parameter | Type | Description |
|------------|------|------------|
| `name` | str | Identifier of the results |
| `data` | str | Folder containing the evaluation meta-dataset |
| `model` | str | Folder containing the model weights |
| `batch_size` | int | Mini-batch size |
| `device` | str | Device (`cpu`, `cuda`, `mps`, etc.) |
| `bootstrap` | int | Number of bootstrap resamples |

A folder is created in `evaluation/` containing:

- The evaluation configuration  
- MSE, bias, and variance results in `.csv` format  

---

# Multivariate Pipeline

The multivariate pipeline is built in a similar way.

## Multivariate Meta-Dataset Generation

The script `src/multi_metadata.py` generates multivariate meta-datasets.

An additional parameter is required:

| Parameter | Type | Description |
|------------|------|------------|
| `dim` | int | Dimension of the multivariate samples |

Available distributions:

```python
["normal", "uniform", "student_t", "dirichlet"]
```

---

## Multivariate Training

Modules used:

- `src/multi_metastat_dataloader.py`  
- `src/multi_metastat_model.py`  

---

## Multivariate Evaluation

```bash
python src/multi_evaluation.py --config config/config_eval.json
```

Additional parameter:

| Parameter | Type | Description |
|------------|------|------------|
| `knn_k` | int | Number of neighbors for the KNN estimator (NPEET baseline) |
