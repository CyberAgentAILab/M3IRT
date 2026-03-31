"""
M²-IRT CAT Experiment Example
==============================

Same workflow as cat_experiment.py but using M²-IRT (inner-product model).
M²-IRT uses scalar Fisher information summed across items,
while M³-IRT uses matrix determinant (D-optimal design).

Run:
    uv run python example_code/cat.py
"""

from pathlib import Path

import pandas as pd

from m3irt.models.m2irt import M2IRT


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

    # ── 2. Create M²-IRT model ────────────────────────────────────────
    model = M2IRT(
        normal_df,
        shuffled_df=shuffled_df,
        max_epochs=500,
        scale_list=[2],
        device="cpu",
    )

    # ── 3. Run CAT ────────────────────────────────────────────────────
    cat_results = model.cat(
        extraction_range=(1, 5),  # evaluate at 1% to 10%(default: (1, 50))
        include_problems=False,  # Generated csv don't include selected problem names(default: False)
        # include_problems=True,            # Generated csv will include selected problem names
        seed=42,
    )

    # ── 4. Analyze results ────────────────────────────────────────────
    print("\n=== Average Correlation by Extracted Percentage ===")
    summary = (
        cat_results.groupby("extracted_percentage")
        .agg(
            {
                "correlation": "mean",
                "shuffle_ratio": "mean",
            }
        )
        .round(4)
    )
    print(summary.to_string())

    # ── 5. Save ───────────────────────────────────────────────────────
    output_path = Path("result/cat_m2irt_example.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cat_results.to_csv(output_path, index=False)
    print(f"\nResults saved to '{output_path}'")


if __name__ == "__main__":
    main()
