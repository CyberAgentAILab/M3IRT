"""
CLI entry point for M3IRT parameter estimation.

Usage:
    m3irt-estimate --config config/estimate/mmmu_m3irt.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from m3irt.utils.config import CATExperimentConfig, load_config


def load_dataframes(config: CATExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load response tables for estimation."""
    normal_df = pd.read_csv(config.dataset.normal_csv, index_col=0)
    shuffled_df = pd.read_csv(config.dataset.shuffled_csv, index_col=0)

    print(f"Normal responses:   {normal_df.shape}")
    print(f"Shuffled responses: {shuffled_df.shape}")
    print()

    return normal_df, shuffled_df


def create_model(
    config: CATExperimentConfig,
    normal_df: pd.DataFrame,
    shuffled_df: pd.DataFrame,
):
    """Instantiate the selected wrapper model."""
    model_class = config.get_cat_wrapper_class()
    return model_class(
        normal_df,
        shuffled_df=shuffled_df,
        lr=config.training.lr,
        max_epochs=config.training.max_epochs,
        batch_size=config.training.batch_size,
        device=config.training.device,
        eps=config.training.eps,
        scale_list=list(config.grid_search.scale_list or []),
    )


def build_theta_dataframe(estimates: dict) -> pd.DataFrame:
    """Convert theta estimates to a tabular format."""
    theta_rows = []
    for model_name, theta in estimates["theta"].items():
        row = {"model": model_name}
        row.update(theta)
        theta_rows.append(row)

    theta_df = pd.DataFrame(theta_rows).set_index("model")
    return theta_df.rename(columns={"theta_synergy": "theta_cross"})


def build_item_dataframe(estimates: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build item-level discrimination and difficulty tables."""
    disc_df = pd.DataFrame(
        {
            "a_base": estimates["discrimination_base"],
            "a_text": estimates["discrimination_text"],
            "a_image": estimates["discrimination_image"],
            "a_cross": estimates["discrimination_synergy"],
        }
    )
    disc_df["a_total"] = disc_df.sum(axis=1)

    item_df = disc_df.copy()
    if "difficulty_base" in estimates:
        diff_df = pd.DataFrame(
            {
                "b_base": estimates["difficulty_base"],
                "b_text": estimates["difficulty_text"],
                "b_image": estimates["difficulty_image"],
                "b_cross": estimates["difficulty_synergy"],
            }
        )
        diff_df["b_full"] = pd.Series(estimates["difficulty_full"])
        item_df[["b_base", "b_text", "b_image", "b_cross", "b_full"]] = diff_df[
            ["b_base", "b_text", "b_image", "b_cross", "b_full"]
        ]
    else:
        diff_df = pd.DataFrame({"b_full": pd.Series(estimates["difficulty_full"])})
        item_df["b_full"] = diff_df["b_full"]

    return disc_df, item_df


def print_summary(theta_df: pd.DataFrame, disc_df: pd.DataFrame, item_df: pd.DataFrame) -> None:
    """Print the same high-signal summary as the example script."""
    diff_sorted = item_df.sort_values("b_full", ascending=False)
    disc_sorted = disc_df.sort_values("a_total", ascending=False)
    has_components = {"b_base", "b_text", "b_image", "b_cross"}.issubset(item_df.columns)

    print("=" * 60)
    print("=== Model Abilities (theta) ===")
    print("=" * 60)
    print(theta_df.round(4).to_string())
    print()

    print("=" * 60)
    print("=== Discrimination Parameters (a) ===")
    print("=" * 60)
    print("Top 10 most discriminating problems:")
    print(disc_sorted.head(10).round(4).to_string())
    print("\nBottom 10 least discriminating problems:")
    print(disc_sorted.tail(10).round(4).to_string())
    print()

    print("=" * 60)
    if has_components:
        print("=== Difficulty Parameters (components) ===")
    else:
        print("=== Difficulty Parameters ===")
    print("=" * 60)
    print("Top 10 hardest problems:")
    print(diff_sorted.head(10).round(4).to_string())
    print("\nTop 10 easiest problems:")
    print(diff_sorted.tail(10).round(4).to_string())
    print()

    print("=" * 60)
    print("=== Summary Statistics ===")
    print("=" * 60)
    print(f"Number of models:   {theta_df.shape[0]}")
    print(f"Number of problems: {item_df.shape[0]}")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="m3irt-estimate",
        description="Train M3IRT or M2IRT and export parameter estimates from a YAML config file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  m3irt-estimate --config config/estimate/mmmu_m3irt.yaml
""",
    )
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to a YAML config file (e.g. config/estimate/mmmu_m3irt.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and validate config without running estimation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    config.validate()
    if args.dry_run:
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
        print(f"  LR:         {config.training.lr}")
        print(f"  Epochs:     {config.training.max_epochs}")
        print(f"  Device:     {config.training.device}")
        print(f"  Scales:     {config.grid_search.scale_list}")
        print(f"  Train pct:  {config.experiment.train_percentage}")
        print(f"  Test pct:   {config.experiment.test_percentage}")
        print(f"  Seed:       {config.experiment.seed}")
        print(f"  Output dir: {config.experiment.output_dir}")
        return

    if config.experiment.cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.experiment.cuda_device)

    normal_df, shuffled_df = load_dataframes(config)
    model = create_model(config, normal_df, shuffled_df)
    model.train(
        train_percentage=config.experiment.train_percentage,
        test_percentage=config.experiment.test_percentage,
        seed=config.experiment.seed if config.experiment.seed is not None else 42,
    )
    print()

    estimates = model.estimate()
    theta_df = build_theta_dataframe(estimates)
    disc_df, item_df = build_item_dataframe(estimates)

    print_summary(theta_df, disc_df, item_df)

    output_dir = Path(config.experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = config.dataset.name
    theta_path = output_dir / f"estimates_{dataset_name}_theta.csv"
    theta_df.to_csv(theta_path)
    print(f"Theta saved to '{theta_path}'")

    item_path = output_dir / f"estimates_{dataset_name}_items.csv"
    item_df.to_csv(item_path)
    print(f"Item parameters saved to '{item_path}'")


if __name__ == "__main__":
    main()
