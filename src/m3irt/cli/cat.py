"""
CLI entry point for M³-IRT CAT experiments.

Usage:
    m3irt-cat --config config/cat/mmmu_m3irt.yaml
"""

from __future__ import annotations

import argparse
import datetime
import os
import random
import sys

import numpy as np
import pandas as pd
import ray
import torch

from m3irt.utils.config import CATExperimentConfig, load_config


def load_dataframes(config: CATExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the normal and shuffled response tables from config."""
    normal_df = pd.read_csv(config.dataset.normal_csv, index_col=0)
    shuffled_df = pd.read_csv(config.dataset.shuffled_csv, index_col=0)

    print(f"Dataset: {config.dataset.name}")
    print(f"  normal_csv:   {config.dataset.normal_csv}  ({normal_df.shape})")
    print(f"  shuffled_csv: {config.dataset.shuffled_csv}  ({shuffled_df.shape})")

    return normal_df, shuffled_df


def create_wrapper_model(
    config: CATExperimentConfig,
    normal_df: pd.DataFrame,
    shuffled_df: pd.DataFrame,
):
    """Instantiate the high-level wrapper used by the simplified CLI CAT flow."""
    WrapperClass = config.get_cat_wrapper_class()
    scale_list = list(config.grid_search.scale_list or [])
    default_scale = float(scale_list[0]) if scale_list else 4.0

    return WrapperClass(
        normal_df,
        shuffled_df=shuffled_df,
        lr=config.training.lr,
        max_epochs=config.training.max_epochs,
        batch_size=config.training.batch_size,
        device=config.training.device,
        eps=config.training.eps,
        update_lr=config.cat.update_lr,
        update_max_epochs=config.cat.update_max_epochs,
        update_patience=config.cat.update_patience,
        theta_max=default_scale,
        difficulty_base_max=default_scale,
        a_scale=default_scale,
        scale_list=scale_list,
    )


def build_output_path(config: CATExperimentConfig) -> str:
    """Create the output CSV path for the experiment."""
    out_dir = config.experiment.output_dir
    os.makedirs(out_dir, exist_ok=True)

    now = datetime.datetime.now()
    stamp = now.strftime("%m%d_%H%M")
    train_suffix = ""
    if config.experiment.train_percentage < 1.0:
        train_suffix = f"_train{config.experiment.train_percentage}"

    return f"{out_dir}/CAT_{config.dataset.name}_{config.model.type}_{stamp}{train_suffix}.csv"


def run_cat_experiment(config: CATExperimentConfig) -> pd.DataFrame:
    """
    Run a CAT experiment through the high-level wrapper API and save CSV output.
    """
    config.validate()

    if config.experiment.seed is not None:
        random.seed(config.experiment.seed)
        np.random.seed(config.experiment.seed)
        torch.manual_seed(config.experiment.seed)

    if config.experiment.cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.experiment.cuda_device)

    normal_df, shuffled_df = load_dataframes(config)
    model = create_wrapper_model(config, normal_df, shuffled_df)

    seed = config.experiment.seed if config.experiment.seed is not None else 42
    extraction_range = tuple(config.cat.extraction_range)

    try:
        results_df = model.cat(
            extraction_range=extraction_range,
            include_problems=config.cat.include_problems,
            train_percentage=config.experiment.train_percentage,
            seed=seed,
        )
    finally:
        if ray.is_initialized():
            ray.shutdown()

    out_path = build_output_path(config)
    results_df.to_csv(out_path, index=False)
    print(f"Results saved to '{out_path}'")

    return results_df


def main():
    parser = argparse.ArgumentParser(
        prog="m3irt-cat",
        description="Run M3-IRT CAT experiments from a YAML config file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  m3irt-cat --config config/cat/mmmu_m3irt.yaml
  m3irt-cat --config config/cat/mathvista_m3irt.yaml

Available model types:
  m3irt      M³-IRT wrapper
  m2irt      Inner-product M²-IRT wrapper
""",
    )
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to a YAML config file (e.g. config/cat/mmmu_m3irt.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and validate config without running the experiment.",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Dry run
    if args.dry_run:
        config.validate()
        try:
            config.get_cat_wrapper_class()
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        print("\n=== Config (dry run) ===")
        print(f"  Dataset:    {config.dataset.name}")
        print(f"  Normal CSV: {config.dataset.normal_csv}")
        print(f"  Shuffle CSV: {config.dataset.shuffled_csv}")
        print(f"  Model:      {config.model.type}")
        print(f"  Grid scales: {config.grid_search.scale_list}")
        print(f"  Extraction range: {config.cat.extraction_range[0]}%–{config.cat.extraction_range[1]}%")
        print(f"  Seed:       {config.experiment.seed}")
        print(f"  Output:     {config.experiment.output_dir}")
        print("\nConfig is valid. Use without --dry-run to execute.")
        return

    # Run
    try:
        run_cat_experiment(config)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
