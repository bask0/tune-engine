# Author: Basil Kraft
# Created: 2025-07-16
# Description: Tuning engine module for managing hyperparameter optimization and cross-validation
#              workflows using Optuna and PyTorch Lightning.

"""
Module for managing hyperparameter tuning and cross-validation with Optuna and PyTorch Lightning.

This module defines the `TuningEngine` class and associated helpers for:
- setting up reproducible experiments,
- managing tuning and cross-validation runs,
- interfacing with PyTorch Lightning's CLI,
- and storing results in Optuna studies.
"""

import sys
import shutil
import warnings
import logging
from pathlib import Path
from os import PathLike
from typing import TYPE_CHECKING, Literal

import optuna
from optuna_integration.pytorch_lightning import PyTorchLightningPruningCallback

from project.core.cli_interface import CLI

if TYPE_CHECKING:
    from optuna import Trial

logger = logging.getLogger("lightning.pytorch")

# Suppress known warnings for cleaner output
warnings.filterwarnings(
    "ignore", message="LightningCLI's args parameter is intended")
warnings.filterwarnings(
    "ignore", message="The 'val_dataloader' does not have many workers which may be a bottleneck.")
warnings.filterwarnings(
    "ignore", message="The 'train_dataloader' does not have many workers which may be a bottleneck.")
warnings.filterwarnings(
    "ignore", message="The 'test_dataloader' does not have many workers which may be a bottleneck.")
warnings.filterwarnings(
    "ignore", message="The 'prediction_dataloader' does not have many workers which may be a bottleneck.")
warnings.filterwarnings(
    "ignore", message="Argument ``multivariate`` is an experimental feature.")

TYPE_MAP = {
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
}


class TuningEngine:
    """
    Engine for hyperparameter tuning and cross-validation using Optuna with PyTorch Lightning.

    This class orchestrates the full lifecycle of hyperparameter optimization and cross-validation
    experiments, including directory management, parameter sampling, study setup, and integration
    with PyTorch Lightning's CLI. It supports both automated hyperparameter search and deterministic
    cross-validation, and manages experiment artifacts such as configuration files, logs, and best models.

    Note that experiment directory and log directory are separated, as logs can be large and don't belong
    into the code base. the log directory is symlinked into the experiment directory for easy access with
    name '[exp_dir]/logs'.

    Directory Structure:
        - `exp_dir`: Main experiment directory.
            - `config.yaml`: Base configuration file.
            - `logs`: Symlink to the logging directory.
        - `log_dir`: Directory for logs and Optuna databases.
            - `tune`: Directory for hyperparameter tuning trials.
                - `optuna.db`: Optuna database for tuning (with study name [EXPERIMENT_NAME]).
                - `trial_000`, ..., `trial_nnn`: Subdirectories for each tuning trial.
            - `xval`: Directory for cross-validation trials.
                - `optuna.db`: Optuna database for cross-validation (with study name [EXPERIMENT_NAME]).
                - `fold_000`, ..., `fold_nnn`: Subdirectories for each cross-validation fold.

    Key Features:
        - Automated setup and cleanup of experiment directories and symlinks.
        - Flexible search space definition and parameter sampling for Optuna studies.
        - Seamless integration with PyTorch Lightning's CLI for model training and evaluation.
        - Support for both hyperparameter tuning (with pruning) and cross-validation (with brute-force sampling).
        - Automatic management and copying of the best configuration file.
        - Robust handling of study creation, loading, and storage using SQLite databases.
        - Utility methods for argument parsing, directory flattening, and symlink creation.

    Usage:
        1. Instantiate with the desired sampler, pruner, search space, and directory paths.
        2. Call `setup()` to prepare the experiment environment.
        3. Use `tune()` to run hyperparameter optimization, or `xval()` for cross-validation.
        4. Access results and artifacts via the returned Optuna study and output directories.

    """

    EXPERIMENT_NAME = "experiment"

    def __init__(
            self,
            sampler: optuna.samplers.BaseSampler,
            pruner: optuna.pruners.BasePruner,
            search_space: dict,
            exp_dir: PathLike | str,
            log_dir: PathLike | str,
            study_name: str = "default_study",
            n_trials: int = 10,
            num_folds: int | None = None,
            monitor: str = 'val_loss',
            skip_tuning: bool = False,
            overwrite: bool = False,
            ):
        """
        Initializes the TuningEngine with the specified configuration.

        Args:
            sampler (optuna.samplers.BaseSampler): The Optuna sampler to use for hyperparameter suggestion.
            pruner (optuna.pruners.BasePruner): The Optuna pruner to use for early stopping of unpromising trials.
            search_space (dict): The hyperparameter search space as a dictionary.
            exp_dir (PathLike | str): Path to the experiment directory where results and configurations will be saved.
            log_dir (PathLike | str): Path to the logging directory for storing logs and Optuna databases.
            study_name (str, optional): Name of the Optuna study. Defaults to "default_study".
            n_trials (int, optional): Number of trials to run for hyperparameter optimization. Defaults to 10.
            num_folds (int | None): Number of folds for cross-validation. Must be specified.
            monitor (str, optional): Metric to monitor for optimization. Defaults to 'val_loss'.
            skip_tuning (bool, optional): If True, skips the tuning process. Defaults to False.
            overwrite (bool, optional): If True, overwrites existing results and configurations. Defaults to False.

        Raises:
            ValueError: If num_folds is not specified.
        """

        if num_folds is None:
            raise ValueError("num_folds must be specified for TuningEngine.")

        self.sampler = sampler
        self.pruner = pruner
        self.search_space = self._flatten_dict(search_space)
        self.exp_dir = Path(exp_dir)
        self.log_dir = Path(log_dir)
        self.study_name = study_name
        self.n_trials = n_trials
        self.num_folds = num_folds
        self.monitor = monitor
        self.skip_tuning = skip_tuning
        self.overwrite = overwrite

        self.tune_dir = self.log_dir / "tune"
        self.xval_dir = self.log_dir / "xval"
        self.tune_db = self.tune_dir / "optuna.db"
        self.xval_db = self.xval_dir / "optuna.db"
        self.best_config_path = self.tune_dir / "best_config.yaml"

    def setup(self, which: Literal["all", "tune", "xval"]) -> None:
        """
        Set up the experiment environment based on the specified mode.

        WARNING: This method will delete existing log directories if `overwrite` is True.

        Args:
            which (Literal["all", "tune", "xval"]): Specifies which parts of the environment to set up.
                - "all": Set up both tuning and cross-validation environments.
                - "tune": Set up only the tuning environment.
                - "xval": Set up only the cross-validation environment.

        Raises:
            ValueError: If 'which' is not one of "all", "tune", or "xval".
            FileNotFoundError: If the experiment directory does not exist.

        Side Effects:
            - Creates or resets directories for tuning and/or cross-validation as needed.
            - Initializes Optuna studies for tuning and/or cross-validation.
            - Creates a symlink for the log directory in the experiment directory.
        """

        if which not in ("all", "tune", "xval"):
            raise ValueError(f"Invalid setup option: {which}. Choose from 'all', 'tune', or 'xval'.")

        # Only exp_dir must exist, others will be created
        if not self.exp_dir.exists():
            raise FileNotFoundError(f"Experiment directory does not exist: {self.exp_dir}")

        if which in ("all", "tune"):
            self._create_dirs_if_needed(self.tune_dir, remove=True)
            self._init_optuna_study(self.tune_db)

        if which in ("all", "xval"):
            self._create_dirs_if_needed(self.xval_dir, remove=True)
            self._init_optuna_study(self.xval_db)

        self._create_symlink(self.log_dir, self.exp_dir / "logs", force=self.overwrite)

    def tune(self) -> optuna.Study:
        """
        Runs the hyperparameter tuning process using Optuna and manages the best configuration.

        If tuning is skipped (`self.skip_tuning` is True), loads and returns an existing Optuna study.
        Otherwise, runs the tuning process, copies the best configuration file to the experiment directory,
        and returns the Optuna study object.

        Returns:
            optuna.Study: The Optuna study object containing all trials and results.
        """

        if self.skip_tuning:
            logger.info("Skipping tuning as skip_tuning is enabled.")
            return optuna.load_study(
                study_name=self.EXPERIMENT_NAME,
                storage=self._get_storage(db_path=self.tune_db),
            )

        study = self.run(is_tune=True)

        best_trial = study.best_trial
        best_log_dir = Path(best_trial.user_attrs["log_dir"])
        best_config_path = best_log_dir / "config.yaml"
        shutil.copy(best_config_path, self.best_config_path)

        return study

    def xval(self) -> optuna.Study:
        """
        Performs cross-validation by running the tuning engine without hyperparameter optimization.

        Returns:
            optuna.Study: The Optuna study object containing the results of the cross-validation run.
        """

        return self.run(is_tune=False)

    def run(self, is_tune: bool = True) -> optuna.Study:
        """
        Runs the Optuna study for hyperparameter tuning or cross-validation.

        Args:
            is_tune (bool, optional): If True, runs hyperparameter tuning using the configured sampler and pruner.
                If False, runs cross-validation using brute-force sampling and no pruning. Defaults to True.

        Returns:
            optuna.Study: The Optuna study object after optimization.

        Raises:
            FileNotFoundError: If `is_tune` is False and the best configuration file does not exist.
        """

        if is_tune:
            sampler = self.sampler
            pruner = self.pruner
            db_path = self.tune_db
        else:
            if not self.best_config_path.exists():
                raise FileNotFoundError(f"Best config file does not exist: {self.best_config_path}")
            sampler = optuna.samplers.BruteForceSampler()
            pruner = optuna.pruners.NopPruner()
            db_path = self.xval_db

        study = optuna.load_study(
            study_name=self.EXPERIMENT_NAME,
            storage=self._get_storage(db_path=db_path),
            sampler=sampler,
            pruner=pruner,
        )
        study.optimize(
            func=lambda trial: self.objective(trial, is_tune),
            n_trials=self.n_trials if is_tune else self.num_folds,
        )
        return study

    def objective(self, trial: 'Trial', is_tune: bool) -> float:
        """
        Objective function for Optuna hyperparameter optimization or cross-validation.

        This function configures and runs a PyTorch Lightning training session using the provided trial's
        hyperparameters. It supports both hyperparameter tuning (is_tune=True) and cross-validation (is_tune=False)
        modes. The function sets up experiment directories, logging, and callbacks, and manages configuration files
        and CLI arguments for the LightningCLI interface.

        During tuning, it attaches a pruning callback for early stopping based on the monitored metric.
        After training, it saves relevant information (best checkpoint path, log directory, and epoch) to the trial's
        user attributes. The function returns the best validation loss achieved during training, as determined by
        the early stopping callback.

        Args:
            trial (Trial): The Optuna trial object containing the current set of hyperparameters.
            is_tune (bool): If True, runs in hyperparameter tuning mode; if False, runs in cross-validation mode.

        Returns:
            float: The best validation loss achieved during the trial.

        Raises:
            AttributeError: If the checkpoint callback or logger is not found in the trainer.
            ValueError: If the early stopping callback is not found in the trainer.
        """

        # Set experiment directory and version for uniqueness
        version = f"trial_{trial.number:03d}" if is_tune else f"fold_{trial.number:03d}"

        if is_tune:
            sample_args = self.sample_parameters(trial=trial)
        else:
            sample_args = self.xval_parameters(trial=trial)

        config_files, cli_args = self._get_cli_config_and_args()

        if not is_tune:
            config_files = [self.best_config_path]

        args = [
            "--trainer.logger.class_path=lightning.pytorch.loggers.TensorBoardLogger",
            f"--trainer.logger.save_dir={self.tune_dir if is_tune else self.xval_dir}",
            f"--trainer.logger.version={version}",
            "--trainer.logger.name=",
        ] + cli_args + sample_args

        if is_tune:
            trainer_defaults = {
                'callbacks': [
                    PyTorchLightningPruningCallback(trial=trial, monitor=self.monitor),
                ],
            }
        else:
            trainer_defaults = {}

        # Combine base config paths and overrides for LightningCLI
        cli = CLI(
            parser_kwargs=dict(
                default_config_files=config_files,
                parser_mode="omegaconf"
            ),
            save_config_kwargs={"overwrite": True},
            trainer_defaults=trainer_defaults,
            # Pass hyperparameters as CLI overrides
            args=args
        )

        # Fit the model
        cli.trainer.fit(model=cli.model, datamodule=cli.datamodule)

        # Save useful info in trial attributes
        try:
            best_checkpoint = cli.trainer.checkpoint_callback.best_model_path  # type: ignore
        except AttributeError:
            raise AttributeError("No checkpoint callback found; cannot retrieve best model path.")

        try:
            log_dir = cli.trainer.logger.log_dir  # type: ignore
        except AttributeError:
            raise AttributeError("No logger found; cannot retrieve log directory.")

        trial.set_user_attr("best_checkpoint", best_checkpoint)
        trial.set_user_attr("log_dir", log_dir)
        trial.set_user_attr("epoch", cli.trainer.current_epoch)

        # Retrieve the best validation loss from early stopping or logs
        if hasattr(cli.trainer, "early_stopping_callback") and cli.trainer.early_stopping_callback is not None:
            val_loss: float = cli.trainer.early_stopping_callback.best_score.item()
        else:
            raise ValueError("No early stopping callback found; cannot retrieve best validation loss.")

        if not is_tune:
            model = type(cli.model).load_from_checkpoint(best_checkpoint)
            cli.trainer.predict(model=model, datamodule=cli.datamodule)

        return val_loss

    def sample_parameters(self, trial: 'Trial') -> list[str]:
        """
        Samples parameters from the defined search space using the provided Optuna trial object.

        For each parameter in the search space, this method selects a value based on the specified
        suggestion method ("loguniform", "uniform", "int", or "categorical") and appends it as a
        command-line argument string (e.g., '--param=value') to the result list. Implemented methods
        include:
        - "loguniform": Samples a float in a logarithmic scale between low and high.
        - "uniform": Samples a float in a linear scale between low and high.
        - "int": Samples an integer between low and high.
        - "categorical": Samples a value from a list of choices, with an optional dtype
          for casting the value (e.g., int, float, str, bool).

        Example search space::

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

        Args:
            trial (Trial): An Optuna Trial object used to suggest parameter values.

        Returns:
            list[str]: A list of command-line argument strings representing the suggested parameters.

        Raises:
            ValueError: If a required 'dtype' is missing for a categorical parameter or if an unsupported
                suggestion method is encountered.
        """

        suggested_params = []

        for key, spec in self.search_space.items():
            suggestion_method = spec["method"]

            if suggestion_method == "loguniform":
                val = trial.suggest_float(key, float(spec["low"]), float(spec["high"]), log=True)
            elif suggestion_method == "uniform":
                val = trial.suggest_float(key, spec["low"], spec["high"])
            elif suggestion_method == "int":
                val = trial.suggest_int(key, spec["low"], spec["high"])
            elif suggestion_method == "categorical":
                if "dtype" not in spec:
                    raise ValueError(f"Missing 'dtype' for categorical search space: {key}")
                cast_fn = TYPE_MAP[spec["dtype"]]
                val = trial.suggest_categorical(key, [cast_fn(val) for val in spec["choices"]])
            else:
                raise ValueError(f"Unsupported suggestion method: {suggestion_method}.")

            suggested_params.append(f'--{key}={val}')

        return suggested_params

    def xval_parameters(self, trial: 'Trial') -> list[str]:
        """
        Suggests the cross-validation fold index for the current trial and formats it as a CLI argument.

        This method uses the provided Optuna trial to select a fold index in the range [0, num_folds - 1].
        The selected fold index is returned as a command-line argument suitable for passing to the data module.

        Args:
            trial (Trial): The Optuna trial object used to suggest the fold index.

        Returns:
            list[str]: A list containing a single CLI argument string specifying the fold index,
                       e.g., ['--data.init_args.fold_idx=2'].
        """

        val = trial.suggest_int("fold_idx", 0, self.num_folds - 1)
        return [f'--data.init_args.fold_idx={val}']

    @staticmethod
    def _get_cli_config_and_args() -> tuple[list[str], list[str]]:
        """
        Parses command-line arguments to separate configuration file paths from other CLI arguments.

        This method scans sys.argv (excluding the script name) and extracts any config file paths
        specified via '--config', '-c', '--config=...', or '-c=...'. All other arguments are returned
        as a separate list. This is useful for distinguishing configuration files from other CLI overrides.

        Returns:
            tuple[list[str], list[str]]:
                - config_files: List of config file paths provided via '--config' or '-c'.
                - args: List of all other CLI arguments.

        Raises:
            ValueError: If '--config' or '-c' is provided without a following file path.
        """

        config_files = []
        args = []

        it = iter(sys.argv[1:])  # Skip script name
        for arg in it:
            if arg.startswith("--config=") or arg.startswith("-c="):
                config_files.append(arg.split("=", 1)[1])
            elif arg in ("--config", "-c"):
                try:
                    config_files.append(next(it))
                except StopIteration:
                    raise ValueError("Expected a file path after '--config' or '-c'")
            else:
                args.append(arg)

        return config_files, args

    def _get_storage(self, db_path: Path, **kwargs) -> optuna.storages.RDBStorage:
        """
        Create and return an Optuna RDBStorage instance for the specified SQLite database path.

        Args:
            db_path (Path): Path to the SQLite database file. The path will be resolved to an absolute path.
            **kwargs: Additional keyword arguments to pass to optuna.storages.RDBStorage.

        Returns:
            optuna.storages.RDBStorage: An Optuna storage backend configured to use the given SQLite database.

        Notes:
            - This method always returns a new RDBStorage instance pointing to the provided database file.
            - The returned storage can be used for creating or loading Optuna studies.
        """

        return optuna.storages.RDBStorage(
            url=f"sqlite:///{db_path.resolve()}",
            **kwargs,
        )

    def _init_optuna_study(self, db_path: Path) -> None:
        """
        Initializes an Optuna study with the specified database path.

        Attempts to create a new Optuna study using the experiment name and storage backend.
        If a study with the same name already exists, the creation is skipped.

        Args:
            db_path (Path): The path to the database for storing the Optuna study.

        Raises:
            Any exceptions raised by Optuna other than DuplicatedStudyError will propagate.
        """

        try:
            optuna.create_study(
                study_name=self.EXPERIMENT_NAME,
                storage=self._get_storage(db_path=db_path),
                direction="minimize",
                load_if_exists=False,
            )
            logger.info("Created new Optuna study.")
        except optuna.exceptions.DuplicatedStudyError:
            logger.info("Optuna study already exists — skipping creation.")

    def _create_dirs_if_needed(self, dir: PathLike | str, remove: bool = False) -> None:
        """
        Creates the specified directory if it does not exist, with optional removal of existing contents.

        Args:
            dir (PathLike | str): The target directory path to create.
            remove (bool, optional): If True, removes the directory and its contents before creation. Defaults to False.

        Raises:
            OSError: If the directory cannot be created or removed.

        Notes:
            - If `remove` is True, the directory and all its contents will be deleted before creation.
            - Parent directories will be created as needed.
        """

        dir = Path(dir)
        if remove:
            self._remove_dir(dir)
        dir.mkdir(parents=True, exist_ok=True)

    def _remove_dir(self, dir: Path) -> None:
        if dir.exists():
            if self.overwrite:
                shutil.rmtree(dir)
            else:
                raise FileExistsError(
                    f"Directory already exists: {dir}. Use overwrite=True or --tuning_engine.overwrite=True "
                    "from the CLI to remove it."
                )

    @staticmethod
    def _create_symlink(source: Path | str, target: Path | str, force: bool = True) -> None:
        """
        Create a symbolic link pointing from `target` to `source`.

        Args:
            source (Path | str): The original file or directory.
            target (Path | str): The symlink to create or replace.
            force (bool): If True, removes any existing file, symlink, or directory at `target`.

        Raises:
            ValueError: If the source does not exist.
        """

        source = Path(source).resolve()
        target = Path(target)

        if not source.exists():
            raise ValueError(f"Source path does not exist: {source}")

        if target.exists() or target.is_symlink():
            if force:
                if target.is_symlink() or target.is_file():
                    target.unlink()
                elif target.is_dir():
                    # Only remove directory if it's a broken symlink or explicitly forced
                    if target.is_symlink():
                        target.unlink()
                    else:
                        raise FileExistsError(f"Target is a non-symlink directory: {target}")
            else:
                raise FileExistsError(f"Target already exists: {target}")

        target.symlink_to(source, target_is_directory=source.is_dir())
        logging.info(f"Created symlink: {target} → {source}")

    def _flatten_dict(self, d: dict, parent_key: str = '', sep: str = '.') -> dict[str, dict]:
        """
        Recursively flattens a nested dictionary, concatenating keys with a separator.

        Args:
            d (dict): The dictionary to flatten.
            parent_key (str, optional): The base key string to prepend to each key. Defaults to ''.
            sep (str, optional): The separator to use between concatenated keys. Defaults to '.'.

        Returns:
            dict[str, dict]: A flattened dictionary where nested keys are joined by the separator.
                             If a value is a dictionary and does not contain the key "method", it is further flattened.
                             Otherwise, the value is included as is.
        """

        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict) and "method" not in v:
                items.extend(self._flatten_dict(d=v, parent_key=new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
