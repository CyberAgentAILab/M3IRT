"""
High-level API for M³-IRT (Multimodal Multidimensional Multimedia IRT).

Usage::

    import pandas as pd
    from m3irt.models.m3irt import M3IRT

    normal_df = pd.read_csv("normal_responses.csv", index_col=0)

    # Fixed hyperparameters (legacy)
    model = M3IRT(normal_df, scale_list=None)
    model.train()
    estimates = model.estimate()

    # With automatic grid search over scale values (default)
    model = M3IRT(normal_df)
    model.train()
    estimates = model.estimate()
    cat_results = model.cat()
"""

from __future__ import annotations

from m3irt.irt_core.m3irt_base import M3IRT_base
from m3irt.models._multimodal_irt_api import BaseMultimodalIRT


class M3IRT(BaseMultimodalIRT):
    """
    High-level wrapper for M³-IRT.

    M³-IRT uses a multi-dimensional ability/discrimination decomposition
    (base, text, image, synergy) implemented in
    :mod:`m3irt.irt_core.m3irt_base`.

    For CAT item selection, Fisher information matrices are summed across
    group items and scored by matrix determinant (D-optimality).

    See :class:`BaseMultimodalIRT` for the full parameter list and
    inherited methods (``train``, ``estimate``, ``cat``).
    """

    _irt_base_class = M3IRT_base
    _fisher_mode = "determinant"
