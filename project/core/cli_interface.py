# Author: Basil Kraft
# Created: 2025-07-16
# Description: CLI interface for configuring and running hyperparameter optimization and cross-validation
#              workflows using Optuna and PyTorch Lightning.

"""
Custom LightningCLI wrapper for configuring and executing model training workflows with Optuna.

This module defines a specialized CLI class (`CLI`) that extends PyTorch Lightning’s `LightningCLI`.
It provides structured argument parsing, dynamic class instantiation (including Optuna samplers/pruners),
and integration of forced callbacks like early stopping and checkpointing. It is specifically designed
to support hyperparameter tuning and cross-validation via a `TuningEngine`.

Main Features:
    - Argument linking between data module and tuning engine (e.g., number of folds).
    - Dynamic instantiation of Optuna components (e.g., sampler, pruner) based on configuration.
    - Registration of early stopping and checkpointing callbacks with sensible defaults.
    - Compatibility with YAML or command-line configuration using `jsonargparse`.

Typical usage:
    cli = CLI()  # Instantiated and used inside a main() function
    cli.tuning_engine.setup(which="all")  # Prepare directories and optuna database.
    cli.tuning_engine.tune()  # Run hyperparameter tuning.
    cli.tuning_engine.xval()  # Run cross-validation.
"""

import importlib
from typing import Any

from jsonargparse import namespace_to_dict

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.cli import LightningCLI

import optuna


class CLI(LightningCLI):
    """
    A custom LightningCLI for configuring and executing model training using Optuna and PyTorch Lightning.

    This class wraps around the PyTorch LightningCLI to provide additional argument parsing,
    callback configuration, and dynamic instantiation of tuning-related components such as
    samplers and pruners.

    Key Features:
        - Adds and links arguments for tuning configuration.
        - Sets up early stopping and model checkpointing callbacks by default.
        - Automatically constructs the TuningEngine using the parsed configuration.
        - Integrates seamlessly with Optuna for tuning and cross-validation experiments.
    """

    def __init__(self, **kwargs):
        """
        Initializes the class with the provided keyword arguments.

        This constructor calls the superclass initializer with specific arguments to set up
        the LightningModule and LightningDataModule, along with additional configuration options.

        Args:
            **kwargs: Arbitrary keyword arguments passed to the superclass initializer.
        """

        super().__init__(
            pl.LightningModule,
            pl.LightningDataModule,
            run=False,
            subclass_mode_model=True,
            subclass_mode_data=True,
            **kwargs
        )

    def add_arguments_to_parser(self, parser):
        """
        Adds custom arguments and argument blocks to the provided parser for configuring the tuning engine,
        search space, and forced callbacks.

        This method first adds base arguments from the superclass, then registers arguments for the
        TuningEngine class under the "tuning_engine" namespace. It links relevant arguments (such as
        number of folds and monitor metrics) between components. It also adds a search space argument
        for hyperparameter tuning, and registers forced callbacks for early stopping and model checkpointing,
        setting their default values and linking their monitored metrics to the tuning engine.

        Args:
            parser: The argument parser object to which arguments will be added.
        """

        # Add LightningCLI args first
        super().add_arguments_to_parser(parser)

        # Avoid circular import
        from project.core.tuning_engine import TuningEngine

        # Register the tuning_engine argument block
        parser.add_class_arguments(
            TuningEngine,
            "tuning_engine"
        )

        parser.link_arguments(source="data.init_args.num_folds", target="tuning_engine.num_folds")

        # Search space configuration
        parser.add_argument(
            "--search_space",
            type=dict[str, Any],
            help="Search space for hyperparameter tuning"
        )

        # Forced callbacks
        parser.add_lightning_class_args(EarlyStopping, "early_stopping_callback")
        parser.link_arguments(source="tuning_engine.monitor", target="early_stopping_callback.monitor")
        parser.set_defaults({
            "early_stopping_callback.patience": 10,
            "early_stopping_callback.mode": 'min',
            "early_stopping_callback.min_delta": 0.0,
        })
        parser.add_lightning_class_args(ModelCheckpoint, "checkpoint_callback")
        parser.link_arguments(source="tuning_engine.monitor", target="checkpoint_callback.monitor")
        parser.set_defaults({
            "checkpoint_callback.mode": 'min',
            "checkpoint_callback.save_last": True,
            "checkpoint_callback.save_top_k": 1,
            "checkpoint_callback.filename": "best",
        })

    def before_instantiate_classes(self):
        """
        Prepares and instantiates the tuning engine and its dependencies before class instantiation.

        This method performs the following steps:
        1. Imports the TuningEngine class to avoid circular imports.
        2. Retrieves and converts the tuning engine configuration from the main config.
        3. Extracts and removes the 'sampler' and 'pruner' configurations from the tuning engine config.
        4. Instantiates the sampler and pruner objects using the provided configurations.
        5. Instantiates the TuningEngine with the created sampler, pruner, and remaining configuration parameters,
           and assigns it to `self.tuning_engine`.

        Raises:
            Any exceptions raised by `self.class_instantiate` or TuningEngine initialization.
        """

        # Currently no-op
        super().before_instantiate_classes()

        # Avoid circular import
        from project.core.tuning_engine import TuningEngine

        tuner_cfg_raw = self.config["tuning_engine"]

        # Convert everything to a real dict
        tuner_cfg = namespace_to_dict(tuner_cfg_raw)

        # Extract nested configs
        sampler_cfg = tuner_cfg.pop("sampler", None)
        pruner_cfg = tuner_cfg.pop("pruner", None)

        sampler: optuna.samplers.BaseSampler = self.class_instantiate(sampler_cfg)
        pruner: optuna.pruners.BasePruner = self.class_instantiate(pruner_cfg)

        self.tuning_engine = TuningEngine(
            sampler=sampler,
            pruner=pruner,
            **tuner_cfg
        )

    def class_instantiate(self, cfg):
        """
        Dynamically instantiates a class from a configuration dictionary.

        Args:
            cfg (dict): Configuration dictionary with the following keys:
                - 'class_path' (str): The full import path to the class (e.g., 'module.submodule.ClassName').
                - 'init_args' (dict, optional): Dictionary of keyword arguments to pass to the class constructor.

        Returns:
            object: An instance of the specified class, initialized with the provided arguments.

        Raises:
            ValueError: If cfg is not a dictionary.
        """

        if not isinstance(cfg, dict):
            raise ValueError("Expected dict with keys 'class_path' and optional 'init_args'.")

        class_path = cfg["class_path"]
        init_args = cfg.get("init_args", {})

        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls(**init_args)
