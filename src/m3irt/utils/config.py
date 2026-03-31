"""
YAML-based configuration loader for M³-IRT CAT experiments.

Usage:
    from m3irt.utils.config import load_config
    config = load_config("config/cat/mmmu_m3irt.yaml")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# ---------------------------------------------------------------------------
# IRT model registry: model type name → (module path, class name)
# ---------------------------------------------------------------------------
IRT_MODEL_REGISTRY: Dict[str, tuple[str, str]] = {
    # (module, class)
    "m3irt": ("m3irt.irt_core.m3irt_base", "M3IRT_base"),
    "m2irt": ("m3irt.irt_core.m2irt_base", "M2IRT_base"),
}

CAT_WRAPPER_REGISTRY: Dict[str, tuple[str, str]] = {
    "m3irt": ("m3irt.models.m3irt", "M3IRT"),
    "m2irt": ("m3irt.models.m2irt", "M2IRT"),
}


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------
@dataclass
class DatasetConfig:
    """Dataset paths and metadata."""

    name: str
    normal_csv: str
    shuffled_csv: str


@dataclass
class ModelConfig:
    """IRT model selection."""

    type: str = "m3irt"


@dataclass
class TrainingConfig:
    """IRT training hyperparameters."""

    lr: float = 0.001
    batch_size: int = 512
    max_epochs: int = 5000
    device: str = "cpu"
    eps: float = 0.001


@dataclass
class GridSearchConfig:
    """Grid search settings for hyperparameter tuning."""

    scale_list: List[int] = field(default_factory=lambda: [2, 4, 8, 16])


@dataclass
class CATConfig_CAT:
    """CAT loop parameters."""

    extraction_range: List[int] = field(default_factory=lambda: [1, 50])
    update_lr: float = 0.0001
    update_max_epochs: int = 1000
    update_patience: int = 50
    include_problems: bool = False


@dataclass
class ExperimentConfig:
    """General experiment settings."""

    seed: Optional[int] = None
    train_percentage: float = 1.0
    test_percentage: float = 0.0
    filter_no_prefix: bool = False
    output_dir: str = "result"
    cuda_device: Optional[int] = 0


@dataclass
class CATExperimentConfig:
    """Top-level config for a CAT experiment."""

    dataset: DatasetConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    grid_search: GridSearchConfig = field(default_factory=GridSearchConfig)
    cat: CATConfig_CAT = field(default_factory=CATConfig_CAT)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    def validate(self) -> None:
        """Validate config values."""
        # Model type
        valid_types = list(IRT_MODEL_REGISTRY.keys())
        if self.model.type not in valid_types:
            raise ValueError(f"Invalid model.type '{self.model.type}'. Choose from: {valid_types}")

        # Data files exist
        if not Path(self.dataset.normal_csv).exists():
            raise FileNotFoundError(f"normal_csv not found: {self.dataset.normal_csv}")
        if not Path(self.dataset.shuffled_csv).exists():
            raise FileNotFoundError(f"shuffled_csv not found: {self.dataset.shuffled_csv}")

        # Train percentage
        if not (0.0 < self.experiment.train_percentage <= 1.0):
            raise ValueError(f"experiment.train_percentage must be in (0, 1], got {self.experiment.train_percentage}")
        if not (0.0 <= self.experiment.test_percentage < 1.0):
            raise ValueError(f"experiment.test_percentage must be in [0, 1), got {self.experiment.test_percentage}")
        if self.experiment.train_percentage + self.experiment.test_percentage > 1.0:
            raise ValueError("experiment.train_percentage + experiment.test_percentage must be <= 1.0")

    def get_irt_model_class(self):
        """Dynamically import and return the IRT model class."""
        import importlib

        module_path, class_name = IRT_MODEL_REGISTRY[self.model.type]
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    def get_cat_wrapper_class(self):
        """Return the high-level wrapper class used by the CLI CAT flow."""
        import importlib

        if self.model.type not in CAT_WRAPPER_REGISTRY:
            supported = ", ".join(CAT_WRAPPER_REGISTRY.keys())
            raise ValueError(f"CLI CAT supports only [{supported}], got '{self.model.type}'.")

        module_path, class_name = CAT_WRAPPER_REGISTRY[self.model.type]
        module = importlib.import_module(module_path)
        return getattr(module, class_name)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
def _dict_to_dataclass(cls, data: Dict[str, Any]):
    """Recursively convert a dict to a nested dataclass."""
    if data is None:
        return cls()
    import typing

    fieldtypes = typing.get_type_hints(cls)
    kwargs = {}
    for key, value in data.items():
        if key in fieldtypes:
            ft = fieldtypes[key]
            if hasattr(ft, "__dataclass_fields__") and isinstance(value, dict):
                kwargs[key] = _dict_to_dataclass(ft, value)
            else:
                kwargs[key] = value
    return cls(**kwargs)


def load_config(path: str) -> CATExperimentConfig:
    """
    Load a CAT experiment config from a YAML file.

    Args:
        path: Path to the YAML config file.

    Returns:
        A validated CATExperimentConfig instance.

    Example:
        >>> config = load_config("config/cat/mmmu_m3irt.yaml")
        >>> print(config.dataset.name)
        mmmu
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError(f"Empty or invalid YAML file: {path}")

    config = _dict_to_dataclass(CATExperimentConfig, raw)
    return config
