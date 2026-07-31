"""Leakage-safe feature engineering for setting-to-loss meta-models.

The source rows deliberately keep settings, derived parameter counts, and
targets separate.  This module only consumes ``setting`` and ``derived`` and
never looks at a target or provenance field.  Four nested feature sets make it
possible to compare a parameter-count heuristic with increasingly expressive
meta-models without changing the data split.

``FeatureEncoder`` is an ordinary scikit-learn transformer.  In particular,
its categorical vocabulary, constant-column mask, and scaling statistics are
all learned in ``fit``.  Putting it inside a model ``Pipeline`` therefore fits
those quantities independently in each cross-validation fold and does not
inspect holdout or external-question covariates.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


FeatureValue: TypeAlias = int | float | str
FeatureDict: TypeAlias = dict[str, FeatureValue]

FEATURE_SETS = ("params", "optimizer_lr", "compact", "full")
_FEATURE_SET_NAMES = frozenset(FEATURE_SETS)
DATASET_CONDITIONING = ("unaware", "id", "description")
_DATASET_CONDITIONING_NAMES = frozenset(DATASET_CONDITIONING)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a UTF-8 JSONL file and require each non-blank line to be an object."""

    source = Path(path)
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {source} at line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(value, dict):
                raise TypeError(
                    f"Expected a JSON object in {source} at line {line_number}, "
                    f"got {type(value).__name__}"
                )
            rows.append(value)
    return rows


def _as_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(value).__name__}")
    return value


def _setting_and_derived(
    example: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Accept an exported row, a target-free example, or a raw setting wrapper."""

    # External-evaluation records store the target-free model input under an
    # ``example`` key.  Supporting that shape here keeps callers from needing
    # a second feature path.
    if "setting" not in example and isinstance(example.get("example"), Mapping):
        return _setting_and_derived(example["example"])

    if "setting" in example:
        setting = _as_mapping(example["setting"], name="example.setting")
        derived_value = example.get("derived", {})
    else:
        # Also accept a setting itself, with an optional adjacent ``derived``
        # key.  The actual candidate setting consists only of these four
        # namespaces; metadata in the wrapper is deliberately ignored.
        setting_keys = ("model", "optimizer", "loss", "budget")
        setting = {key: example[key] for key in setting_keys if key in example}
        derived_value = example.get("derived", {})

    if not derived_value and isinstance(example.get("features"), Mapping):
        # Exported rows normally have ``derived``.  This narrow fallback is for
        # sanitized setting-only exports; it reads exactly one approved
        # covariate rather than trusting the whole pre-flattened feature map.
        flattened = example["features"]
        if "derived.log_total_params" in flattened:
            derived_value = {
                "log_total_params": flattened["derived.log_total_params"]
            }
    derived = _as_mapping(derived_value, name="example.derived")
    return setting, derived


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric, got bool")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric, got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


def _positive_log(value: Any, *, name: str, base: float = math.e) -> float:
    number = _finite_float(value, name=name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive, got {number!r}")
    return math.log(number, base)


def _log_total_params(derived: Mapping[str, Any]) -> float:
    supplied_log = derived.get("log_total_params")
    if supplied_log is None:
        supplied_log = derived.get("derived.log_total_params")
    total = derived.get("total_params")
    if total is None:
        total = derived.get("derived.total_params")

    if supplied_log is None and total is None:
        raise KeyError(
            "example.derived must contain log_total_params or total_params"
        )
    if supplied_log is None:
        return _positive_log(total, name="derived.total_params")

    log_value = _finite_float(supplied_log, name="derived.log_total_params")
    if total is not None:
        expected = _positive_log(total, name="derived.total_params")
        if not math.isclose(log_value, expected, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError(
                "derived.log_total_params is inconsistent with derived.total_params"
            )
    return log_value


def _optimizer_lr_features(setting: Mapping[str, Any]) -> FeatureDict:
    optimizer = _as_mapping(setting.get("optimizer"), name="setting.optimizer")
    optimizer_type = optimizer.get("type")
    if not isinstance(optimizer_type, str) or not optimizer_type:
        raise ValueError("setting.optimizer.type must be a non-empty string")
    log_lr = _positive_log(
        optimizer.get("lr"),
        name="setting.optimizer.lr",
        base=10.0,
    )
    return {
        "optimizer.type": optimizer_type,
        "optimizer.log10_lr": log_lr,
        # A numeric key per optimizer is the linear type x log(lr)
        # interaction after DictVectorizer expands the dictionary.
        f"interaction.optimizer_x_log10_lr.{optimizer_type}": log_lr,
    }


def _optimizer_details(setting: Mapping[str, Any]) -> FeatureDict:
    optimizer = _as_mapping(setting.get("optimizer"), name="setting.optimizer")
    features: FeatureDict = {}
    for key in ("weight_decay", "momentum"):
        if key in optimizer and optimizer[key] is not None:
            features[f"optimizer.{key}"] = _finite_float(
                optimizer[key], name=f"setting.optimizer.{key}"
            )

    betas = optimizer.get("betas")
    if betas is not None:
        if not isinstance(betas, (list, tuple)) or len(betas) != 2:
            raise ValueError("setting.optimizer.betas must contain exactly two values")
        features["optimizer.beta1"] = _finite_float(
            betas[0], name="setting.optimizer.betas[0]"
        )
        features["optimizer.beta2"] = _finite_float(
            betas[1], name="setting.optimizer.betas[1]"
        )
    features["optimizer.has_momentum"] = int("momentum" in optimizer)
    features["optimizer.has_betas"] = int(betas is not None)
    return features


def _budget_and_loss_features(setting: Mapping[str, Any]) -> FeatureDict:
    features: FeatureDict = {}
    budget_value = setting.get("budget")
    if budget_value is not None:
        budget = _as_mapping(budget_value, name="setting.budget")
        if "total_samples_seen" in budget:
            features["budget.log_total_samples_seen"] = _positive_log(
                budget["total_samples_seen"],
                name="setting.budget.total_samples_seen",
            )
        if "batch_size" in budget:
            features["budget.log_batch_size"] = _positive_log(
                budget["batch_size"], name="setting.budget.batch_size"
            )

    loss_value = setting.get("loss")
    if loss_value is not None:
        loss = _as_mapping(loss_value, name="setting.loss")
        loss_id = loss.get("loss_id")
        if loss_id is not None:
            if not isinstance(loss_id, str) or not loss_id:
                raise ValueError("setting.loss.loss_id must be a non-empty string")
            features["loss.loss_id"] = loss_id
    return features


def _transition_count(values: list[Any]) -> int:
    return sum(left != right for left, right in zip(values, values[1:]))


def _mlp_features(model: Mapping[str, Any]) -> FeatureDict:
    features: FeatureDict = {}
    if "depth" in model:
        features["model.depth"] = _finite_float(
            model["depth"], name="setting.model.depth"
        )
    if "width" in model:
        features["model.log2_width"] = _positive_log(
            model["width"], name="setting.model.width", base=2.0
        )
    if "input_dim" in model:
        features["model.log2_input_dim"] = _positive_log(
            model["input_dim"], name="setting.model.input_dim", base=2.0
        )
    if "residual" in model:
        features["model.residual"] = int(bool(model["residual"]))

    activations_value = model.get("activations", [])
    if not isinstance(activations_value, (list, tuple)):
        raise TypeError("setting.model.activations must be a list")
    activations = list(activations_value)
    if activations:
        if any(not isinstance(value, str) or not value for value in activations):
            raise ValueError("Every setting.model.activations value must be a string")
        counts = Counter(activations)
        n_layers = len(activations)
        features["model.activation_layers"] = n_layers
        features["model.activation_unique_count"] = len(counts)
        features["model.activation_transitions"] = _transition_count(activations)
        for activation, count in sorted(counts.items()):
            features[f"model.activation_count.{activation}"] = count
            features[f"model.activation_fraction.{activation}"] = count / n_layers

    layer_norm_value = model.get("layer_norm", [])
    if not isinstance(layer_norm_value, (list, tuple)):
        raise TypeError("setting.model.layer_norm must be a list")
    layer_norm = [bool(value) for value in layer_norm_value]
    if layer_norm:
        count = sum(layer_norm)
        features["model.layer_norm_layers"] = len(layer_norm)
        features["model.layer_norm_count"] = count
        features["model.layer_norm_fraction"] = count / len(layer_norm)
        features["model.layer_norm_any"] = int(count > 0)
        features["model.layer_norm_all"] = int(count == len(layer_norm))
        features["model.layer_norm_transitions"] = _transition_count(layer_norm)
    return features


def _transformer_features(model: Mapping[str, Any]) -> FeatureDict:
    features: FeatureDict = {}
    numeric: dict[str, float] = {}
    for key in (
        "d_model",
        "d_ff",
        "num_layers",
        "num_heads",
        "context_length",
        "vocab_size",
    ):
        if key in model:
            value = _finite_float(model[key], name=f"setting.model.{key}")
            if value <= 0.0:
                raise ValueError(f"setting.model.{key} must be positive")
            numeric[key] = value
            features[f"model.{key}"] = value

    if "d_ff" in numeric and "d_model" in numeric:
        features["model.d_ff_over_d_model"] = (
            numeric["d_ff"] / numeric["d_model"]
        )
    if "d_model" in numeric and "num_heads" in numeric:
        features["model.head_dim"] = numeric["d_model"] / numeric["num_heads"]
    if "num_layers" in numeric and "d_model" in numeric:
        features["model.layers_x_d_model"] = (
            numeric["num_layers"] * numeric["d_model"]
        )
    return features


def _architecture_features(setting: Mapping[str, Any]) -> FeatureDict:
    model = _as_mapping(setting.get("model"), name="setting.model")
    model_type = model.get("type")
    if not isinstance(model_type, str) or not model_type:
        raise ValueError("setting.model.type must be a non-empty string")
    features: FeatureDict = {"model.type": model_type}
    if model_type == "mlp":
        features.update(_mlp_features(model))
    elif model_type == "transformer_lm":
        features.update(_transformer_features(model))
    return features


def _flatten_json(value: Any, prefix: str, output: FeatureDict) -> None:
    if isinstance(value, Mapping):
        for key in sorted(value, key=str):
            child = f"{prefix}.{key}" if prefix else str(key)
            _flatten_json(value[key], child, output)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _flatten_json(item, f"{prefix}[{index}]", output)
        return
    if isinstance(value, bool):
        output[prefix] = int(value)
        return
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        output[prefix] = _finite_float(value, name=f"setting.{prefix}")
        return
    if isinstance(value, str):
        output[prefix] = value
        return
    if value is None:
        return
    raise TypeError(
        f"Unsupported setting value at {prefix!r}: {type(value).__name__}"
    )


def feature_dict(
    example: Mapping[str, Any],
    feature_set: str,
    *,
    include_parameter_count: bool = True,
    dataset_conditioning: str = "unaware",
) -> FeatureDict:
    """Build one of the four target-free feature dictionaries.

    The sets are nested in expressive power:

    - ``params``: natural log of the exact generated-model parameter count;
    - ``optimizer_lr``: parameter count, optimizer, log learning rate, and
      optimizer-by-learning-rate interaction;
    - ``compact``: engineered optimizer, budget, loss, and architecture
      summaries suitable for simple linear rules;
    - ``full``: compact features plus every raw setting scalar/position.

    Raw and trainable parameter counts are intentionally never emitted.  The
    sole size feature is ``derived.log_total_params``.
    """

    if feature_set not in _FEATURE_SET_NAMES:
        known = ", ".join(FEATURE_SETS)
        raise ValueError(f"Unknown feature_set {feature_set!r}; expected one of: {known}")
    if dataset_conditioning not in _DATASET_CONDITIONING_NAMES:
        known = ", ".join(DATASET_CONDITIONING)
        raise ValueError(
            f"Unknown dataset_conditioning {dataset_conditioning!r}; expected one of: {known}"
        )
    if feature_set == "params" and not include_parameter_count:
        raise ValueError("The params feature set requires parameter count")
    source = _as_mapping(example, name="example")
    setting, derived = _setting_and_derived(source)
    features: FeatureDict = {}
    context = source.get("dataset_context")
    if dataset_conditioning != "unaware":
        context = _as_mapping(context, name="example.dataset_context")
        if dataset_conditioning == "id":
            dataset_id = context.get("dataset_id")
            if not isinstance(dataset_id, str) or not dataset_id:
                raise ValueError("example.dataset_context.dataset_id must be a non-empty string")
            features["dataset.id"] = dataset_id
        else:
            description = _as_mapping(
                context.get("description"), name="example.dataset_context.description"
            )
            _flatten_json(description, "dataset.description", features)
    if include_parameter_count:
        features["derived.log_total_params"] = _log_total_params(derived)
    if feature_set == "params":
        return features

    features.update(_optimizer_lr_features(setting))
    if feature_set == "optimizer_lr":
        return dict(sorted(features.items()))

    features.update(_optimizer_details(setting))
    features.update(_budget_and_loss_features(setting))
    features.update(_architecture_features(setting))
    if feature_set == "full":
        raw_features: FeatureDict = {}
        _flatten_json(setting, "", raw_features)
        features.update(raw_features)

    # Sorting is not required by DictVectorizer, but makes feature snapshots and
    # interpretation artifacts deterministic and easy to diff.
    return dict(sorted(features.items()))


def _coerce_examples(values: Any) -> list[Mapping[str, Any]]:
    if isinstance(values, Mapping):
        raw_examples: list[Any] = [values]
    elif isinstance(values, np.ndarray):
        if values.ndim == 0:
            raw_examples = [values.item()]
        elif values.ndim == 1:
            raw_examples = values.tolist()
        elif values.ndim == 2 and values.shape[1] == 1:
            raw_examples = values[:, 0].tolist()
        else:
            raise ValueError(
                "FeatureEncoder expects a one-dimensional object array of rows, "
                f"got shape {values.shape}"
            )
    else:
        if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
            raise TypeError("FeatureEncoder input must be an iterable of mappings")
        raw_examples = list(values)

    examples: list[Mapping[str, Any]] = []
    for index, value in enumerate(raw_examples):
        if not isinstance(value, Mapping):
            raise TypeError(
                f"FeatureEncoder row {index} must be a mapping, "
                f"got {type(value).__name__}"
            )
        examples.append(value)
    return examples


class FeatureEncoder(TransformerMixin, BaseEstimator):
    """Fit-on-train dict vectorization, variance filtering, and scaling.

    ``__init__`` only records ``feature_set``, as required for cloning inside
    scikit-learn search and cross-validation utilities.  Learned objects use
    trailing-underscore names and are naturally serializable with ``joblib``.
    """

    def __init__(
        self,
        feature_set: str = "full",
        include_parameter_count: bool = True,
        dataset_conditioning: str = "unaware",
    ) -> None:
        self.feature_set = feature_set
        self.include_parameter_count = include_parameter_count
        self.dataset_conditioning = dataset_conditioning

    def fit(self, X: Any, y: Any = None) -> FeatureEncoder:
        del y
        examples = _coerce_examples(X)
        if not examples:
            raise ValueError("FeatureEncoder.fit requires at least one row")
        dictionaries = [
            feature_dict(
                row,
                self.feature_set,
                include_parameter_count=self.include_parameter_count,
                dataset_conditioning=self.dataset_conditioning,
            )
            for row in examples
        ]

        # Keep these as locals until every stage succeeds.  A failed refit does
        # not leave a mixture of newly and previously fitted state behind.
        vectorizer = DictVectorizer(sparse=False, sort=True)
        vectorized = vectorizer.fit_transform(dictionaries)
        variance_threshold = VarianceThreshold(threshold=0.0)
        reduced = variance_threshold.fit_transform(vectorized)
        scaler = StandardScaler()
        scaler.fit(reduced)

        self.vectorizer_ = vectorizer
        self.variance_threshold_ = variance_threshold
        self.scaler_ = scaler
        self.n_output_features_ = int(reduced.shape[1])
        return self

    def transform(self, X: Any) -> np.ndarray:
        check_is_fitted(
            self,
            ("vectorizer_", "variance_threshold_", "scaler_"),
        )
        examples = _coerce_examples(X)
        if not examples:
            return np.empty((0, self.n_output_features_), dtype=np.float64)
        dictionaries = [
            feature_dict(
                row,
                self.feature_set,
                include_parameter_count=self.include_parameter_count,
                dataset_conditioning=self.dataset_conditioning,
            )
            for row in examples
        ]
        vectorized = self.vectorizer_.transform(dictionaries)
        reduced = self.variance_threshold_.transform(vectorized)
        return np.asarray(self.scaler_.transform(reduced), dtype=np.float64)

    @property
    def feature_names(self) -> list[str]:
        """Names of post-variance-filter columns, in output matrix order."""

        check_is_fitted(self, ("vectorizer_", "variance_threshold_"))
        raw_names = self.vectorizer_.get_feature_names_out()
        return raw_names[self.variance_threshold_.get_support()].tolist()

    def get_feature_names_out(
        self,
        input_features: Any = None,
    ) -> np.ndarray:
        """Return output names using scikit-learn's transformer convention."""

        del input_features
        return np.asarray(self.feature_names, dtype=object)
