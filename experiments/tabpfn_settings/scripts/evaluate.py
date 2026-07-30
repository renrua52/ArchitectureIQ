#!/usr/bin/env python3
"""Cross-validated baselines + TabPFN on a settings table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT_EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_EXP))

from src_tabpfn_settings.features import FEATURE_COLUMNS  # noqa: E402
from src_tabpfn_settings.metrics import regression_report  # noqa: E402


def _split_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    cat, num = [], []
    for c in cols:
        if pd.api.types.is_bool_dtype(df[c]) or df[c].dtype == object:
            cat.append(c)
        else:
            # treat mostly-string columns as categorical
            if df[c].dropna().map(lambda x: isinstance(x, str)).any():
                cat.append(c)
            else:
                num.append(c)
    return cat, num


def _sklearn_pipeline(cat: list[str], num: list[str], model: str) -> Pipeline:
    transformers = []
    if num:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num,
            )
        )
    if cat:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat,
            )
        )
    pre = ColumnTransformer(transformers)
    if model == "ridge":
        est = Ridge(alpha=1.0)
    elif model == "hgb":
        est = HistGradientBoostingRegressor(max_depth=4, max_iter=200, random_state=0)
    else:
        raise ValueError(model)
    return Pipeline([("pre", pre), ("est", est)])


def _tabpfn_predict(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    *,
    device: str,
    model_version: str = "v2.5",
) -> np.ndarray:
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    version_map = {
        "v2": ModelVersion.V2,
        "v2.5": ModelVersion.V2_5,
        "v2.6": ModelVersion.V2_6,
        "v3": ModelVersion.V3,
    }
    if model_version not in version_map:
        raise SystemExit(f"Unknown --tabpfn-version {model_version}")
    # Prefer create_default_for_version so China hosts can use cached non-v3 ckpts
    # without HuggingFace gated-repo auth for TabPFN-3.
    reg = TabPFNRegressor.create_default_for_version(version_map[model_version], device=device)
    reg.fit(X_train, y_train)
    return np.asarray(reg.predict(X_test), dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--table",
        type=Path,
        default=ROOT_EXP / "artifacts" / "xor_table.csv",
    )
    parser.add_argument("--target", default="mean_test_ce")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--tabpfn-version",
        default="v2.5",
        choices=["v2", "v2.5", "v2.6", "v3"],
        help="TabPFN checkpoint family (v3 is HF-gated; v2.5 is cached on the lab A100).",
    )
    parser.add_argument("--skip-tabpfn", action="store_true")
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT_EXP / "artifacts" / "p0_report.json",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.table)
    if args.target not in df.columns:
        raise SystemExit(f"Missing target {args.target}; columns={list(df.columns)}")
    df = df.dropna(subset=[args.target]).copy()
    y = df[args.target].to_numpy(dtype=float)
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[feature_cols].copy()
    # Drop columns that are entirely missing (common when one family is absent).
    X = X.dropna(axis=1, how="all")
    feature_cols = list(X.columns)
    # normalize bools to object categories for TabPFN friendliness
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].map({True: "true", False: "false", None: None})

    lower_is_better = not args.target.endswith("accuracy")
    cat, num = _split_columns(X)
    kf = KFold(n_splits=min(args.folds, len(df)), shuffle=True, random_state=args.seed)

    results: dict[str, object] = {
        "table": str(args.table),
        "target": args.target,
        "n_rows": int(len(df)),
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "models": {},
    }

    # mean baseline
    mean_pred = np.full_like(y, fill_value=float(np.mean(y)))
    results["models"]["train_mean"] = regression_report(y, mean_pred, lower_is_better=lower_is_better)

    for name in ("ridge", "hgb"):
        pipe = _sklearn_pipeline(cat, num, name)
        pred = cross_val_predict(pipe, X, y, cv=kf)
        results["models"][name] = regression_report(y, pred, lower_is_better=lower_is_better)

    if not args.skip_tabpfn:
        preds = np.zeros_like(y, dtype=float)
        for fold, (tr, te) in enumerate(kf.split(X)):
            print(f"TabPFN fold {fold + 1}/{kf.get_n_splits()} n_train={len(tr)} n_test={len(te)}")
            preds[te] = _tabpfn_predict(
                X.iloc[tr],
                y[tr],
                X.iloc[te],
                device=args.device,
                model_version=args.tabpfn_version,
            )
        results["models"]["tabpfn"] = regression_report(y, preds, lower_is_better=lower_is_better)
        results["tabpfn_version"] = args.tabpfn_version

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results["models"], indent=2))
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
