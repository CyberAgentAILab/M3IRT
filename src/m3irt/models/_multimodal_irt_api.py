"""
Shared base class for M²-IRT and M³-IRT high-level APIs.

This module contains :class:`BaseMultimodalIRT`, which consolidates all
common logic (data preparation, grid-search training, estimation, CAT
experiment) used by both :class:`M2IRT` and :class:`M3IRT`.

Grid-search and CAT steps are parallelized with **Ray** following the
same pattern as ``mfi_mshuffle.py`` / ``run_cat.py``:

  * **Phase 1 — Grid Search** : For each (deletion-model × scale) pair,
    train with validation and evaluate ROC-AUC.  All combinations run as
    independent Ray tasks.
  * **Phase 2 — CAT** : For each deletion model, train with the best
    scale found in Phase 1 then run the adaptive-testing loop.  Each
    model's CAT is an independent Ray task.

Subclasses must set:
  1. ``_irt_base_class`` — the underlying IRT ``nn.Module`` to instantiate.
  2. ``_fisher_mode``  — ``"determinant"`` (M³-IRT) or ``"scalar_sum"``
     (M²-IRT), controlling how per-item Fisher information is aggregated
     for CAT item selection.
"""

from __future__ import annotations

import abc
import random
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import ray
import torch

from m3irt.utils.cat_utils import (
    _ray_eval_scale_for_cat,
    _ray_eval_scale_for_train,
    _ray_run_cat_for_model,
    extract_group,
)
from m3irt.utils.contami_judge import is_contaminated
from m3irt.utils.masks import (
    create_balanced_mask,
    create_balanced_mask_with_fixed_test,
)

# ---------------------------------------------------------------------------
# Prefix tags for shuffled problem variants
# ---------------------------------------------------------------------------
PREFIXES = ["<no_image>", "<no_question>", "<no_info>"]


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------
class BaseMultimodalIRT(abc.ABC):
    """
    Abstract base class shared by M²-IRT and M³-IRT wrappers.

    Subclasses must set:
      - ``_irt_base_class``: the concrete IRT model class.
      - ``_fisher_mode``: ``"determinant"`` or ``"scalar_sum"``.
    """

    # Subclasses must assign these.
    _irt_base_class = None  # type: type
    _fisher_mode: str = None  # "determinant" or "scalar_sum"

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        normal_df: pd.DataFrame,
        shuffled_df: pd.DataFrame = None,
        *,
        lr: float = 0.001,
        max_epochs: int = 5000,
        batch_size: int = 512,
        device: str = "cpu",
        eps: float = 0.001,
        update_lr: float = 0.0001,
        update_max_epochs: int = 100,
        update_patience: int = 50,
        theta_max: float = 4.0,
        difficulty_base_max: float = 4.0,
        a_scale: float = 4.0,
        scale_list: Optional[List[float]] = [2, 4, 8, 16],
    ):
        # ── Store raw DataFrames ──────────────────────────────────────
        self.normal_df = normal_df.copy()
        self.shuffled_df = shuffled_df.copy() if shuffled_df is not None else None

        if shuffled_df is not None:
            self.merged_df = pd.merge(
                normal_df,
                shuffled_df,
                left_index=True,
                right_index=True,
            )
            self._low_quality_columns = set(shuffled_df.columns.tolist())
        else:
            self.merged_df = normal_df.copy()
            self._low_quality_columns = set()

        self.problems: List[str] = self.merged_df.columns.tolist()
        self.models: List[str] = self.merged_df.index.tolist()
        self.full_resp: np.ndarray = self.merged_df.values.astype(np.float32)

        # ── Hyperparameters ───────────────────────────────────────────
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.device = device
        self.eps = eps
        self.update_lr = update_lr
        self.update_max_epochs = update_max_epochs
        self.update_patience = update_patience
        self.theta_max = theta_max
        self.difficulty_base_max = difficulty_base_max
        self.a_scale = a_scale
        self.scale_list = scale_list

        # ── Internal state ────────────────────────────────────────────
        self._irt_model = None
        self._trained: bool = False
        self._best_scale: Optional[float] = None
        self._best_estimates: Optional[Dict] = None

    # ------------------------------------------------------------------
    # Internal: build an IRT model
    # ------------------------------------------------------------------
    def _build_irt(
        self,
        mask: np.ndarray,
        theta_max: float,
        difficulty_base_max: float,
        a_scale: float,
    ):
        return self._irt_base_class(
            response_data=self.full_resp,
            train_mask=mask,
            student_names=self.models,
            test_names=self.problems,
            lr=self.lr,
            batch_size=self.batch_size,
            max_epochs=self.max_epochs,
            device=self.device,
            eps=self.eps,
            theta_max=theta_max,
            difficulty_base_max=difficulty_base_max,
            difficulty_other_max=difficulty_base_max,
            a_scale=a_scale,
        )

    # ------------------------------------------------------------------
    # Shared kwargs for Ray workers
    # ------------------------------------------------------------------
    def _ray_irt_kwargs(self) -> Dict:
        """Hyperparameter dict shared by all Ray grid-search tasks."""
        return dict(
            lr=self.lr,
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            device=self.device,
            eps=self.eps,
        )

    def _ray_update_kwargs(self) -> Dict:
        """Update-step kwargs for CAT Ray tasks."""
        return dict(
            update_lr=self.update_lr,
            update_max_epochs=self.update_max_epochs,
            update_patience=self.update_patience,
        )

    # ------------------------------------------------------------------
    # Public API: train
    # ------------------------------------------------------------------
    def train(
        self,
        train_percentage: float = 0.95,
        test_percentage: float = 0.0,
        val_percentage: float = 0.05,
        seed: int = 42,
    ):
        """
        Train the IRT model.

        When ``scale_list`` has > 1 entries, performs a **parallel** grid
        search (via Ray) to select the best scale, then re-trains with it.

        Parameters
        ----------
        train_percentage : float
            Fraction of valid responses used for training (0, 1].
        test_percentage : float
            Fraction of responses held out for test evaluation.
        val_percentage : float
            Fraction for validation (ignored when grid search is active).
        seed : int
            Random seed.

        Returns
        -------
        self
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if self.scale_list and len(self.scale_list) > 1:
            self._train_with_grid_search(
                train_percentage=train_percentage,
                test_percentage=test_percentage,
                seed=seed,
            )
        else:
            self._train_single(
                train_percentage=train_percentage,
                test_percentage=test_percentage,
                val_percentage=val_percentage,
                seed=seed,
                theta_max=self.theta_max,
                difficulty_base_max=self.difficulty_base_max,
                a_scale=self.a_scale,
            )

        self._trained = True
        return self

    def _train_single(
        self,
        train_percentage: float,
        test_percentage: float,
        val_percentage: float,
        seed: int,
        theta_max: float,
        difficulty_base_max: float,
        a_scale: float,
    ):
        """Train a single model with specified hyperparameters."""
        mask, _ = create_balanced_mask(
            response_data=self.full_resp,
            test_percentage=test_percentage,
            item_names=self.problems,
            seed=seed,
            val_percentage=val_percentage,
        )

        if train_percentage < 1.0:
            mask, _ = create_balanced_mask_with_fixed_test(
                mask=mask,
                response_data=self.full_resp,
                train_percentage=train_percentage,
                item_names=self.problems,
                seed=seed,
            )
        else:
            mask[mask == -1] = 1

        self._irt_model = self._build_irt(mask, theta_max, difficulty_base_max, a_scale)
        self._irt_model.fit()

    def _train_with_grid_search(
        self,
        train_percentage: float,
        test_percentage: float,
        seed: int,
    ):
        """
        Parallel grid search over scale_list to find the best
        hyperparameters (validation ROC-AUC), then re-train with best.
        """
        gs_val_percentage = 0.1

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        print("=== Grid Search over scale_list (Ray parallel) ===")
        print(f"    scales={self.scale_list}")

        full_resp_ref = ray.put(self.full_resp)
        kw = self._ray_irt_kwargs()

        tasks = []
        for scale in self.scale_list:
            tasks.append(
                _ray_eval_scale_for_train.remote(
                    self._irt_base_class,
                    full_resp_ref,
                    self.problems,
                    self.models,
                    scale,
                    train_percentage,
                    test_percentage,
                    gs_val_percentage,
                    seed,
                    **kw,
                )
            )

        results = ray.get(tasks)

        best = max(results, key=lambda r: r["roc_auc"])
        best_scale = best["scale"]
        best_roc_auc = best["roc_auc"]

        print(f"=== Best scale={best_scale} (validation roc_auc={best_roc_auc:.4f}) ===")

        self._best_scale = best_scale

        # Re-train with best hyperparameters (no validation split)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        mask, _ = create_balanced_mask(
            response_data=self.full_resp,
            test_percentage=test_percentage,
            item_names=self.problems,
            seed=seed,
            val_percentage=0.0,
        )
        if train_percentage < 1.0:
            mask, _ = create_balanced_mask_with_fixed_test(
                mask=mask,
                response_data=self.full_resp,
                train_percentage=train_percentage,
                item_names=self.problems,
                seed=seed,
            )
        else:
            mask[mask == -1] = 1

        self._irt_model = self._build_irt(mask, best_scale, best_scale, best_scale)
        self._irt_model.fit()

        self._best_estimates = self._irt_model.get_estimates()
        self.theta_max = best_scale
        self.difficulty_base_max = best_scale
        self.a_scale = best_scale

    # ------------------------------------------------------------------
    # Public API: estimate
    # ------------------------------------------------------------------
    def estimate(self) -> Dict:
        """
        Return IRT parameter estimates from the trained model.

        Raises
        ------
        RuntimeError
            If :meth:`train` has not been called.
        """
        if not self._trained:
            raise RuntimeError("Model has not been trained yet. Call train() first.")
        return self._irt_model.get_estimates()

    # ------------------------------------------------------------------
    # Public API: cat
    # ------------------------------------------------------------------
    def cat(
        self,
        extraction_range: tuple = (1, 50),
        include_problems: bool = False,
        train_percentage: float = 1.0,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Run a Computerized Adaptive Testing (CAT) experiment.

        Follows the ``mfi_mshuffle.py`` / ``run_cat.py`` pattern:

        * **Phase 1** — Per-deletion-model grid search (Ray parallel)
          over ``scale_list`` to find the best scale for each model.
        * **Phase 2** — CAT loop for every model (Ray parallel), each
          using its best scale from Phase 1.

        Parameters
        ----------
        extraction_range : tuple of (int, int)
            Percentage range for milestones. Default: (1, 50).
        include_problems : bool
            Include administered problem list in results.
        train_percentage : float
            Fraction of data used for training. Default: 1.0.
        seed : int
            Random seed. Default: 42.

        Returns
        -------
        pd.DataFrame
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        lo, hi = extraction_range
        milestone_percents = [i / 100.0 for i in range(lo, hi + 1)]

        # ── Phase 1: Per-model grid search ────────────────────────
        best_params_per_model: Dict[str, Dict[str, float]] = {}
        if self.scale_list and len(self.scale_list) > 1:
            print("=== CAT Phase 1: Per-model grid search (Ray parallel) ===")
            best_params_per_model = self._grid_search_per_model(
                train_percentage=train_percentage,
                seed=seed,
            )

        # ── Phase 2: CAT loop for each model (Ray parallel) ──────
        print("=== CAT Phase 2: Running CAT experiments (Ray parallel) ===")

        full_resp_ref = ray.put(self.full_resp)
        merged_df_ref = ray.put(self.merged_df)
        normal_df_ref = ray.put(self.normal_df)
        kw = self._ray_irt_kwargs()
        update_kw = self._ray_update_kwargs()

        cat_tasks = []
        for model_name in self.models:
            if model_name in best_params_per_model:
                bp = best_params_per_model[model_name]
                tm = bp["theta_max"]
                dbm = bp["difficulty_base_max"]
                asc = bp["a_scale"]
            else:
                tm = self.theta_max
                dbm = self.difficulty_base_max
                asc = self.a_scale

            cat_tasks.append(
                _ray_run_cat_for_model.remote(
                    self._irt_base_class,
                    full_resp_ref,
                    merged_df_ref,
                    normal_df_ref,
                    self.problems,
                    self.models,
                    model_name,
                    tm,
                    dbm,
                    asc,
                    milestone_percents,
                    train_percentage,
                    include_problems,
                    seed,
                    self._fisher_mode,
                    self._low_quality_columns,
                    **kw,
                    **update_kw,
                )
            )

        print(f"    Launched {len(cat_tasks)} CAT tasks")
        cat_results_nested = ray.get(cat_tasks)

        all_records = [rec for sub in cat_results_nested for rec in sub]
        result_df = pd.DataFrame(all_records)
        if not include_problems and "problems" in result_df.columns:
            result_df = result_df.drop(columns=["problems"])

        return result_df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_low_quality(self, problem_name: str) -> bool:
        """Check low-quality status using the legacy contamination rule."""
        return is_contaminated(problem_name)

    def _grid_search_per_model(
        self,
        train_percentage: float,
        seed: int,
    ) -> Dict[str, Dict[str, float]]:
        """
        Per-model grid search for CAT (leave-one-out with validation).

        All (model × scale) combinations run as Ray tasks in parallel,
        matching the ``eval_one_config`` pattern in ``mfi_mshuffle.py``.
        """
        gs_val_percentage = 0.2

        group_names = [extract_group(col) for col in self.normal_df.columns]
        unique_groups = list(set(group_names))
        random.seed(seed)
        n_val_groups = min(100, len(unique_groups))
        selected_groups = random.sample(unique_groups, n_val_groups)
        val_items = [col for col in self.merged_df.columns if extract_group(col) in selected_groups]

        full_resp_ref = ray.put(self.full_resp)
        val_items_ref = ray.put(val_items)
        kw = self._ray_irt_kwargs()

        # Launch all (model × scale) tasks
        tasks = []
        for model_name in self.models:
            for scale in self.scale_list:
                tasks.append(
                    _ray_eval_scale_for_cat.remote(
                        self._irt_base_class,
                        full_resp_ref,
                        self.problems,
                        self.models,
                        model_name,
                        scale,
                        val_items_ref,
                        train_percentage,
                        gs_val_percentage,
                        seed,
                        **kw,
                    )
                )

        print(
            f"    Launched {len(tasks)} grid-search tasks ({len(self.models)} models × {len(self.scale_list)} scales)"
        )

        results = ray.get(tasks)

        # Aggregate best per model
        best_params: Dict[str, Dict[str, float]] = {}
        best_auc: Dict[str, float] = {}

        for r in results:
            mn = r["model_name"]
            roc_auc = r["roc_auc"]
            if mn not in best_auc or roc_auc > best_auc[mn]:
                best_auc[mn] = roc_auc
                best_params[mn] = {
                    "theta_max": r["scale"],
                    "difficulty_base_max": r["scale"],
                    "a_scale": r["scale"],
                }

        print("=== Best params per model ===")
        for mn in self.models:
            if mn in best_params:
                print(f"  {mn}: scale={best_params[mn]['theta_max']} (roc_auc={best_auc[mn]:.4f})")

        return best_params
