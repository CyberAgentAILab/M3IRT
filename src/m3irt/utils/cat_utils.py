from __future__ import annotations

import random
from typing import Dict, List

import numpy as np
import pandas as pd
import ray
import torch
from scipy.stats import spearmanr

from m3irt.utils.contami_judge import is_contaminated
from m3irt.utils.masks import (
    create_balanced_mask,
    create_balanced_mask_with_fixed_test,
)

# ---------------------------------------------------------------------------
# Prefix tags for shuffled problem variants
# ---------------------------------------------------------------------------
PREFIXES = ["<no_image>", "<no_question>", "<no_info>"]


def extract_group(col_name: str) -> str:
    """Strip known prefixes to get the base group name."""
    for p in PREFIXES:
        if col_name.startswith(p):
            return col_name[len(p) :]
    return col_name


# ---------------------------------------------------------------------------
# Fisher scoring helper (used inside Ray tasks)
# ---------------------------------------------------------------------------


def _score_fisher(
    fisher: Dict[str, object],
    group_items: List[str],
    mode: str,
) -> float:
    """
    Aggregate per-item Fisher information for a problem group.

    Parameters
    ----------
    fisher : dict
        Mapping from problem name to Fisher info (matrix or scalar).
    group_items : list of str
        Problem names in the group.
    mode : str
        ``"determinant"`` — sum matrices, then take ``det()`` (M³-IRT).
        ``"scalar_sum"`` — sum scalar values (M²-IRT).
    """
    if mode == "determinant":
        ref_shape = next(iter(fisher.values())).shape
        mat = sum(fisher.get(p, np.zeros(ref_shape)) for p in group_items)
        return float(np.linalg.det(mat))
    else:  # "scalar_sum"
        return float(sum(fisher.get(p, 0.0) for p in group_items))


def _build_milestone_step_map(
    total_units: int,
    milestone_percents: List[float],
) -> Dict[int, List[int]]:
    """Map CAT steps to target integer percentages for CSV logging.

    For each target percentage, choose the smallest step whose actual
    percentage rounded to two decimals is closest to that target.
    """
    if total_units <= 0:
        return {}

    step_map: Dict[int, List[int]] = {}
    target_percentages = sorted({int(round(pct * 100)) for pct in milestone_percents})

    for target in target_percentages:
        best_step = min(
            range(1, total_units + 1),
            key=lambda step: (
                abs(round((100.0 * step) / total_units, 2) - target),
                step,
            ),
        )
        step_map.setdefault(best_step, []).append(target)

    return step_map


# ---------------------------------------------------------------------------
# Ray remote workers
# ---------------------------------------------------------------------------


@ray.remote(num_cpus=1)
def _ray_eval_scale_for_train(
    irt_base_class,
    full_resp: np.ndarray,
    problems: List[str],
    models: List[str],
    scale: float,
    train_percentage: float,
    test_percentage: float,
    gs_val_percentage: float,
    seed: int,
    *,
    lr: float,
    max_epochs: int,
    batch_size: int,
    device: str,
    eps: float,
) -> Dict:
    """Evaluate a single scale value for train() grid search (Ray task)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    mask, _ = create_balanced_mask(
        response_data=full_resp,
        test_percentage=test_percentage,
        item_names=problems,
        seed=seed,
        val_percentage=gs_val_percentage,
    )

    if train_percentage < 1.0:
        mask, _ = create_balanced_mask_with_fixed_test(
            mask=mask,
            response_data=full_resp,
            train_percentage=train_percentage,
            item_names=problems,
            seed=seed,
        )
    else:
        mask[mask == -1] = 1

    irt = irt_base_class(
        response_data=full_resp,
        train_mask=mask,
        student_names=models,
        test_names=problems,
        lr=lr,
        batch_size=batch_size,
        max_epochs=max_epochs,
        device=device,
        eps=eps,
        theta_max=scale,
        difficulty_base_max=scale,
        difficulty_other_max=scale,
        a_scale=scale,
    )
    irt.fit()
    metrics = irt.evaluate_validation()
    roc_auc = metrics.get("roc_auc", 0.0)

    print(f"    scale={scale} → validation roc_auc={roc_auc:.4f}")
    return {"scale": scale, "roc_auc": roc_auc}


@ray.remote(num_cpus=1)
def _ray_eval_scale_for_cat(
    irt_base_class,
    full_resp: np.ndarray,
    problems: List[str],
    models: List[str],
    model_name: str,
    scale: float,
    val_items: List[str],
    train_percentage: float,
    gs_val_percentage: float,
    seed: int,
    *,
    lr: float,
    max_epochs: int,
    batch_size: int,
    device: str,
    eps: float,
) -> Dict:
    """Evaluate (deletion-model, scale) for CAT grid search (Ray task)."""
    from m3irt.utils.masks import create_validation_mask_for_items

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    del_idx = models.index(model_name)

    train_mask, _ = create_validation_mask_for_items(
        response_data=full_resp,
        item_names=problems,
        val_item_names=val_items,
        val_percentage=gs_val_percentage,
        seed_init=seed,
    )
    train_mask[train_mask == 1] = -1
    train_mask, _ = create_balanced_mask_with_fixed_test(
        mask=train_mask,
        response_data=full_resp,
        train_percentage=train_percentage,
        item_names=problems,
        seed=seed,
    )
    train_mask[del_idx, :] = -1

    irt = irt_base_class(
        response_data=full_resp,
        train_mask=train_mask,
        student_names=models,
        test_names=problems,
        lr=lr,
        batch_size=batch_size,
        max_epochs=max_epochs,
        device=device,
        eps=eps,
        theta_max=scale,
        difficulty_base_max=scale,
        difficulty_other_max=scale,
        a_scale=scale,
    )
    irt.fit()
    metrics = irt.evaluate_validation()
    roc_auc = metrics.get("roc_auc", 0.0)

    return {
        "model_name": model_name,
        "scale": scale,
        "roc_auc": roc_auc,
    }


@ray.remote(num_cpus=1)
def _ray_run_cat_for_model(
    irt_base_class,
    full_resp: np.ndarray,
    merged_df: pd.DataFrame,
    normal_df: pd.DataFrame,
    problems: List[str],
    models: List[str],
    model_name: str,
    theta_max: float,
    difficulty_base_max: float,
    a_scale: float,
    milestone_percents: List[float],
    train_percentage: float,
    include_problems: bool,
    seed: int,
    fisher_mode: str,
    low_quality_columns: set,
    *,
    lr: float,
    max_epochs: int,
    batch_size: int,
    device: str,
    eps: float,
    update_lr: float,
    update_max_epochs: int,
    update_patience: int,
) -> List[Dict]:
    """Run the CAT loop for a single deletion model (Ray task).

    Follows the same logic as ``run_deletion_cat_experiment_milestones``
    in ``mfi_mshuffle.py``.
    """
    # ── Build train mask (model excluded) ─────────────────────────
    train_mask = np.ones_like(full_resp, dtype=np.float32)
    train_mask[train_mask == 1] = -1
    train_mask, _ = create_balanced_mask_with_fixed_test(
        mask=train_mask,
        response_data=full_resp,
        train_percentage=train_percentage,
        item_names=problems,
        seed=seed,
    )

    try:
        del_idx = models.index(model_name)
    except ValueError:
        print(f"  Warning: model '{model_name}' not found, skipping.")
        return []

    train_mask[del_idx, :] = -1

    # ── Problem groups ────────────────────────────────────────────
    problem_groups: Dict[str, List[str]] = {}
    for p in problems:
        base = extract_group(p)
        problem_groups.setdefault(base, []).append(p)

    all_units = list(problem_groups.keys())
    total = len(all_units)
    milestone_step_map = _build_milestone_step_map(total, milestone_percents)
    target_steps = sorted(milestone_step_map.keys())

    # ── Train IRT model ───────────────────────────────────────────
    print(f"[{model_name}] IRT training started (scale={a_scale})")
    irt = irt_base_class(
        response_data=full_resp,
        train_mask=train_mask,
        student_names=models,
        test_names=problems,
        lr=lr,
        batch_size=batch_size,
        max_epochs=max_epochs,
        device=device,
        eps=eps,
        theta_max=theta_max,
        difficulty_base_max=difficulty_base_max,
        difficulty_other_max=difficulty_base_max,
        a_scale=a_scale,
    )
    irt.fit()

    # ── CAT loop ──────────────────────────────────────────────────
    administered: set = set()
    admin_list: List[str] = []
    records: List[Dict] = []
    step = 0

    gt_columns = list(normal_df.columns)

    while step < max(target_steps):
        fisher = irt.compute_item_fisher_information(model_name)
        scores: Dict[str, float] = {}

        for unit in all_units:
            if unit in administered:
                continue
            scores[unit] = _score_fisher(fisher, problem_groups[unit], fisher_mode)

        if not scores:
            break

        chosen = max(scores, key=scores.get)
        administered.add(chosen)
        admin_list.extend(problem_groups[chosen])
        step += 1

        for idx, p in enumerate(problems):
            if extract_group(p) in administered:
                train_mask[del_idx, idx] = 1

        irt.set_train_mask(train_mask)

        irt.update_single_theta(
            model_name=model_name,
            problem_names=admin_list,
            lr=update_lr,
            max_epochs=update_max_epochs,
            patience=update_patience,
        )

        if step in milestone_step_map:
            eval_items = list(admin_list)

            subset_means = merged_df[eval_items].mean(axis=1)
            overall_gt = normal_df[gt_columns].mean(axis=1)
            corr, _ = spearmanr(subset_means.rank(), overall_gt.rank())

            shuffle_ratio = sum(is_contaminated(p) for p in eval_items) / max(1, len(eval_items))
            actual_percentage = round((100.0 * step) / total, 1)

            for target_percentage in milestone_step_map[step]:
                record = {
                    "model": model_name,
                    "extracted_percentage": actual_percentage,
                    "correlation": corr,
                    "shuffle_ratio": shuffle_ratio,
                }
                if include_problems:
                    record["problems"] = list(eval_items)

                records.append(record)
                print(
                    f"  [{model_name}] {actual_percentage}% "
                    f"corr={corr:.4f}, "
                    f"gamma(shuffled problem ratio)={shuffle_ratio:.4f}"
                )

    return records
