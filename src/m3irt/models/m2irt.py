"""
High-level API for M2-IRT (Multi-Modal IRT).

Usage::

    import pandas as pd
    from m3irt.models.m2irt import M2IRT

    normal_df = pd.read_csv("normal_responses.csv", index_col=0)

    # Fixed hyperparameters (legacy)
    model = M2IRT(normal_df, scale_list=None)
    model.train()
    estimates = model.estimate()

    # With automatic grid search over scale values (default)
    model = M2IRT(normal_df)
    model.train()
    estimates = model.estimate()
    cat_results = model.cat()
"""

from __future__ import annotations

from m3irt.irt_core.m2irt_base import M2IRT_base
from m3irt.models._multimodal_irt_api import BaseMultimodalIRT


class M2IRT(BaseMultimodalIRT):
    """
    High-level wrapper for M²-IRT (Inner-Product Multimodal IRT).

    M²-IRT uses an inner-product formulation for combining multimodal
    ability and discrimination parameters, implemented in
    :mod:`m3irt.irt_core.m2irt_base`.

    For CAT item selection, scalar Fisher information values are summed
    across group items.

    See :class:`BaseMultimodalIRT` for the full parameter list and
    inherited methods (``train``, ``estimate``, ``cat``).
    """

    _irt_base_class = M2IRT_base
    _fisher_mode = "scalar_sum"
