"""
M³-IRT Estimate Example
=========================

This example demonstrates how to:
1. Load response data (normal + shuffled)
2. Train an M³-IRT model with grid search
3. Retrieve and analyze parameter estimates (θ, a, b)
4. Evaluate prediction performance on held-out test data
5. Export estimates to CSV for further analysis

Run:
    uv run python example_code/estimate.py
"""

from pathlib import Path

import pandas as pd

from m3irt.models.m3irt import M3IRT
from m3irt.models.m2irt import M2IRT  # noqa: F401  (used in commented example below)


def main():
    # ── 1. Load data ──────────────────────────────────────────────────
    normal_df = pd.read_csv(
        "responses/mmmu/normal_mmmu.csv",
        index_col=0,
    )
    shuffled_df = pd.read_csv(
        "responses/mmmu/shuffled_mmmu.csv",
        index_col=0,
    )
    print(f"Normal responses:   {normal_df.shape}")
    print(f"Shuffled responses: {shuffled_df.shape}")
    print()

    # ── 2. Create and train M³-IRT model ──────────────────────────────
    model = M3IRT(
        normal_df,
        shuffled_df=shuffled_df,
        lr=0.001,
        max_epochs=5000,
        scale_list=[2, 4, 8, 16],  # grid search over scales
        device="cpu",
    )

    # If you want to use M2IRT:
    # model = M2IRT(
    #     normal_df,
    #     shuffled_df=shuffled_df,
    #     lr=0.001,
    #     max_epochs=5000,
    #     scale_list=[2, 4, 8, 16],
    #     device="cpu",
    # )

    # Train with grid search (uses validation split internally)
    model.train(
        train_percentage=0.95,
        test_percentage=0.05,
        seed=42,
    )
    print()

    # ── 3. Get parameter estimates ────────────────────────────────────
    estimates = model.estimate()

    # --- 3a. Theta (model abilities) ---
    print("=" * 60)
    print("=== Model Abilities (θ) ===")
    print("=" * 60)
    theta_rows = []
    for model_name, theta in estimates["theta"].items():
        row = {"model": model_name}
        row.update(theta)
        theta_rows.append(row)
    theta_df = pd.DataFrame(theta_rows).set_index("model")
    theta_df = theta_df.rename(columns={"theta_synergy": "theta_cross"})
    print(theta_df.round(4).to_string())
    print()

    # --- 3b. Discrimination parameters (a) ---
    print("=" * 60)
    print("=== Discrimination Parameters (a) ===")
    print("=" * 60)
    disc_df = pd.DataFrame(
        {
            "a_base": estimates["discrimination_base"],
            "a_text": estimates["discrimination_text"],
            "a_image": estimates["discrimination_image"],
            "a_cross": estimates["discrimination_synergy"],
        }
    )
    disc_df["a_total"] = disc_df.sum(axis=1)
    disc_df = disc_df.sort_values("a_total", ascending=False)
    print("Top 10 most discriminating problems:")
    print(disc_df.head(10).round(4).to_string())
    print("\nBottom 10 least discriminating problems:")
    print(disc_df.tail(10).round(4).to_string())
    print()

    # --- 3c. Difficulty parameters (b) ---
    print("=" * 60)
    print("=== Difficulty Parameters (components) ===")
    print("=" * 60)
    diff_df = pd.DataFrame(
        {
            "b_base": estimates["difficulty_base"],
            "b_text": estimates["difficulty_text"],
            "b_image": estimates["difficulty_image"],
            "b_cross": estimates["difficulty_synergy"],
        }
    )
    diff_df["b_full"] = pd.Series(estimates["difficulty_full"])
    diff_sorted = diff_df.sort_values("b_full", ascending=False)
    print("Top 10 hardest problems:")
    print(diff_sorted.head(10).round(4).to_string())
    print("\nTop 10 easiest problems:")
    print(diff_sorted.tail(10).round(4).to_string())
    print()

    # --- 3d. Summary statistics ---
    print("=" * 60)
    print("=== Summary Statistics ===")
    print("=" * 60)
    print(f"Number of models:   {theta_df.shape[0]}")
    print(f"Number of problems: {len(estimates['difficulty_full'])}")
    print()
    print("Theta statistics:")
    print(theta_df.describe().round(4).to_string())
    print()
    print("Discrimination statistics:")
    print(disc_df.describe().round(4).to_string())
    print()
    print("Difficulty statistics:")
    print(diff_df[["b_base", "b_text", "b_image", "b_cross"]].describe().round(4).to_string())
    print()

    # ── 4. Export estimates to CSV ────────────────────────────────────
    output_dir = "result"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save theta
    theta_path = f"{output_dir}/estimates_theta.csv"
    theta_df.to_csv(theta_path)
    print(f"Theta saved to '{theta_path}'")

    # Save item parameters (discrimination + difficulty)
    item_df = disc_df.copy()
    item_df[["b_base", "b_text", "b_image", "b_cross"]] = diff_df[["b_base", "b_text", "b_image", "b_cross"]]
    item_path = f"{output_dir}/estimates_items.csv"
    item_df.to_csv(item_path)
    print(f"Item parameters saved to '{item_path}'")

    print("\nDone!")


if __name__ == "__main__":
    main()
