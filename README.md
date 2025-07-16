# Optuna Tuning Engine

**Author:** Basil Kraft  
**Created:** July 16, 2025

A modular, CLI-based framework for hyperparameter tuning and cross-validation using [Optuna](https://optuna.org/) and [PyTorch Lightning](https://lightning.ai/).

This project supports:
- Clean experiment tracking via directory structure and logging
- Configurable search spaces and callbacks
- Reproducible CLI-driven experiments

---

## 🔧 Key Features

- 🚀 **Fully configurable via command line or YAML**
- 🎯 **Supports both tuning and k-fold cross-validation**
- 💾 **Auto-generated best configs and checkpoints**
- 📂 **Symlinked logs for clean separation of data and code**
- 🔌 **Compatible with LightningCLI & Optuna integrations**
- 🔍 **Pruning, checkpointing, early stopping — all built-in**

---

## To-dos

- Parallel execution using Optuna

---

## 🧩 Using this project as a blueprint

This repository is designed to serve as a **blueprint** for hyperparameter tuning pipelines. You are expected to:

- Add your own models under `models/`
- Implement a custom datamodule under `data/`
- Define your tuning/search configurations under `experiments/`

To get started **and retain compatibility with updates**, please fork the repository.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/bask0/tune-engine.git
cd tune-engine
```

### 2. Create the environment

I recommend using **conda** to install dependencies.

```bash
conda env create -f environment.yml
conda activate tuning_engine
```

### 3. Define your experiment

Create a base configuration file and tuning configuration, e.g.:

```bash
experiments/
  base_config.yaml     # Fixed model/data params
  exp1/
    config.yaml        # Tuning-specific overrides
```

### 4. Run tuning + cross-validation

```bash
python tune.py \
  --config=experiments/base_config.yaml \
  --config=experiments/exp1/config.yaml \
  --tuning_engine.overwrite=True
```

This will:

1. Initialize the tuning and xval directories  
2. Tune the model using Optuna (parallel trials)  
3. Run k-fold cross-validation with the best configuration  

---

## 🧠 Project Structure

```bash
.
├── tune.py                # Entry point
├── engine/
│   ├── cli_interface.py   # Custom LightningCLI wrapper
│   └── tuning_engine.py   # Core tuning logic
├── experiments/           # Experiment configuration
│   ├── base_config.yaml   # Overarching base config
│   ├── exp0               # An experiment
│   │   ├── logs/          # (symlinked) logs (`log_dir`)
│   │   └── config.yaml    # Experiment config
├── models/                # Lightning models
│   └── ...                # Add custom models
├── data/                  # Lightning datamodule
│   └── ...                # Add custom datamodule
└── environment.yml        # Full conda + pip setup
log_dir
├── tune                   # Hyperparameter tuning log dir
│   ├── trial_000          # Triel 0 directory.
│   └── ...                # Up to trial `tune_engine.n_trials`
├── xval                   # Cross validation log dir
│   ├── fold_000           # Fold 0 directory.
──  └── ...                # Up to fold `data.num_folds`
```

---

## 📦 Dependencies

Main libraries include:

- `optuna`, `optuna-integration`  
- `pytorch`, `torch-geometric`, `lightning`  
- `scikit-learn`, `numpy`, `pandas`, `xarray`, `dask`  
- `jsonargparse[signatures]`, `omegaconf`, `tensorboard`  
- `jupyterlab-optuna`, `matplotlib`, `plotly`, `geopandas`, ...

> See [environment.yml](./environment.yml) for a full list.

This is just the usual libraries I need; most are not required for the framework.

---

## 🔬 Customizing Experiments

### Use Optuna samplers/pruners

```yaml
tuning_engine:
  sampler:
    class_path: optuna.samplers.TPESampler
    init_args:
      seed: 42
  pruner:
    class_path: optuna.pruners.MedianPruner
    init_args:
      n_warmup_steps: 5
```

### Add your own model and data

```yaml
model:
  class_path: models.lit_model.LitModel
  init_args:
    hidden_size: 32

data:
  class_path: data.datamodule.MNISTDataModule
  init_args:
    batch_size: 64
    fold_idx: 0
    num_folds: 3
```

### Define a search space

The structure must follow the configuration hierarchy (e.g., `model.hidden_size`), but the `init_args` key is optional and can be omitted here.

```yaml
tuning_engine:
  search_space:
    model:
      learning_rate:
        method: loguniform
        low: 1e-5
        high: 1e-1
      hidden_size:
        method: categorical
        dtype: int
        choices: [16, 32, 64]
```

---

## 💡 Tips

- ⛔ Logs and checkpoints are **not** committed to Git. A symlink (`logs/`) is created from experiment directory to log directory.  
- ✅ Use `--tuning_engine.overwrite=True` to reset experiments.  
- 🔁 Parallelism is currently not implemented. It can be handled via `n_jobs` in Optuna, but this would require MySQL or Postgres databases.
