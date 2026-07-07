from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from architecture_iq.families.base import DatasetFamily
    from architecture_iq.models.base import ModelFamily

_DATASET_FAMILIES: dict[str, DatasetFamily] = {}
_MODEL_TYPES: dict[str, ModelFamily] = {}


def register_dataset_family(family: DatasetFamily) -> None:
    _DATASET_FAMILIES[family.name] = family


def register_model_type(model: ModelFamily) -> None:
    _MODEL_TYPES[model.name] = model


def get_dataset_family(name: str) -> DatasetFamily:
    if name not in _DATASET_FAMILIES:
        raise KeyError(f"Unknown dataset family: {name}")
    return _DATASET_FAMILIES[name]


def get_model_type(name: str) -> ModelFamily:
    if name not in _MODEL_TYPES:
        raise KeyError(f"Unknown model type: {name}")
    return _MODEL_TYPES[name]


def list_dataset_families() -> list[str]:
    return list(_DATASET_FAMILIES)


def list_model_types() -> list[str]:
    return list(_MODEL_TYPES)


def _register_all() -> None:
    from architecture_iq.families.bigram_lm import BigramLmFamily
    from architecture_iq.families.multivariate_regression import MultivariateRegressionFamily
    from architecture_iq.families.univariate_regression import UnivariateRegressionFamily
    from architecture_iq.models.mlp import MlpModelFamily
    from architecture_iq.models.transformer_lm import TransformerLmModelFamily

    register_dataset_family(UnivariateRegressionFamily())
    register_dataset_family(MultivariateRegressionFamily())
    register_dataset_family(BigramLmFamily())
    register_model_type(MlpModelFamily())
    register_model_type(TransformerLmModelFamily())


_BOOTSTRAPPED = False


def ensure_registries() -> None:
    global _BOOTSTRAPPED
    if not _BOOTSTRAPPED:
        _register_all()
        _BOOTSTRAPPED = True
