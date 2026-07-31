#!/usr/bin/env python
"""Direct winner-prediction meta-model.

Instead of regressing log(loss) and then picking the argmin, this trains a
classifier whose input is the *pairwise difference* of two candidates'
feature vectors and whose target is "does candidate A beat candidate B".

For three-choice evaluation, we score each triple by running all 3 pairwise
comparisons (A vs B, A vs C, B vs C) and selecting the candidate with the
most "wins". This is a Bradley-Terry / pairwise-ranking approach.

Why this is better suited to tree models:
  - The target is binary (win/lose), not continuous loss.
  - Trees handle piecewise decision boundaries naturally.
  - No assumption of a smooth loss surface.

Setting: baseline = one model per dataset (same as dataset_pooled_id).
The classifier trains on all pairs within each dataset's train rows.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from tools.meta_model_study.wide import load_snapshot
from tools.meta_model_study.features import FeatureEncoder
from tools.meta_model_study.significant_recheck import load_candidate_summary


def check_triple_significance(
    summaries: list[dict],
    metric: str,
    gap_min: float = 0.05,
    win_rate_min: float = 0.7,
    use_non_overlap: bool = True,
    higher_is_better: bool = False,
) -> bool:
    """Inline copy of validate_significance logic for 3 candidates."""
    if any(s.get("excluded") for s in summaries):
        return False
    mean_key = f"mean_{metric}"
    std_key = f"std_{metric}"
    final_key = f"final_{metric}"
    means = np.array([s[mean_key] for s in summaries], dtype=np.float64)
    stds = np.array([s[std_key] for s in summaries], dtype=np.float64)
    if not np.all(np.isfinite(means)):
        return False
    order = np.argsort(means)
    if higher_is_better:
        order = order[::-1]
    winner = int(order[0])
    runner_up = int(order[1])
    gap = float(abs(means[runner_up] - means[winner]))
    if gap < gap_min:
        return False
    n_seeds = len(summaries[0]["seed_results"])
    wins = 0
    for seed_i in range(n_seeds):
        vals = []
        for s in summaries:
            sr = s["seed_results"][seed_i]
            vals.append(float("inf") if sr["failed"] else sr[final_key])
        vals_arr = np.array(vals)
        seed_order = np.argsort(vals_arr)
        if higher_is_better:
            seed_order = seed_order[::-1]
        if int(seed_order[0]) == winner:
            wins += 1
    if wins / n_seeds < win_rate_min:
        return False
    if use_non_overlap:
        if means[winner] + stds[winner] >= means[runner_up] - stds[runner_up]:
            return False
    return True


def build_pairwise_dataset(
    rows: list[dict],
    summaries_by_fp: dict[str, dict],
    metric: str,
    encoder: FeatureEncoder,
) -> tuple[np.ndarray, np.ndarray]:
    """Build pairwise (diff, label) training data — vectorized.

    For each pair (i, j): x = features(i) - features(j), y = 1 if i beats j.
    Symmetric pairs added for balance.
    """
    n = len(rows)
    if n < 2:
        return np.empty((0, 0)), np.empty(0)

    X_all = encoder.transform(rows)  # (n, d)
    # use mean_loss directly (family-agnostic, lower=better); fallback to metric if needed
    means = np.array([r["target"]["mean_loss"] for r in rows])

    # vectorized: all pairs via upper-triangular indices
    i_idx, j_idx = np.triu_indices(n, k=1)
    diffs = X_all[i_idx] - X_all[j_idx]  # (n_pairs, d)
    labels = (means[i_idx] < means[j_idx]).astype(np.float64)

    # add symmetric: -diff with flipped label
    X_pairs = np.vstack([diffs, -diffs])
    y_pairs = np.concatenate([labels, 1.0 - labels])
    return X_pairs, y_pairs


def predict_triple_winner(
    triple_features: np.ndarray,  # (3, d)
    model: BaseEstimator,
) -> int:
    """Predict winner of a triple via pairwise votes.

    For each pair, model predicts P(A beats B). Candidate with highest
    total vote count wins.
    """
    votes = np.zeros(3)
    for i, j in combinations(range(3), 2):
        diff = triple_features[i] - triple_features[j]
        p_i_beats_j = model.predict_proba(diff.reshape(1, -1))[0, 1]
        if p_i_beats_j >= 0.5:
            votes[i] += 1
        else:
            votes[j] += 1
    return int(np.argmax(votes))


def make_classifier(name: str, seed: int) -> BaseEstimator:
    if name == "logistic":
        return LogisticRegression(max_iter=5000, C=1.0, random_state=seed)
    elif name == "extra_trees_clf":
        return ExtraTreesClassifier(
            n_estimators=300,
            max_depth=16,
            min_samples_leaf=5,
            max_features=0.7,
            random_state=seed,
            n_jobs=1,
        )
    elif name == "random_forest_clf":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=5,
            max_features=0.7,
            random_state=seed,
            n_jobs=1,
        )
    elif name == "gradient_boosting_clf":
        return GradientBoostingClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            random_state=seed,
        )
    elif name == "mlp_clf":
        return MLPClassifier(
            hidden_layer_sizes=(128, 64),
            alpha=0.01,
            max_iter=2000,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=seed,
        )
    else:
        raise ValueError(f"unknown classifier: {name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--snapshot-manifest", default="artifacts/wide_v2_full30_gt_snapshot.json"
    )
    ap.add_argument("--output-root", required=True)
    ap.add_argument(
        "--classifiers",
        default="logistic,extra_trees_clf,random_forest_clf,gradient_boosting_clf,mlp_clf",
    )
    ap.add_argument("--feature-set", default="full")
    ap.add_argument("--gap-min", type=float, default=0.05)
    ap.add_argument("--win-rate-min", type=float, default=0.7)
    ap.add_argument("--use-non-overlap", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=20260714)
    ap.add_argument(
        "--scope",
        choices=["dataset", "global"],
        default="dataset",
        help="dataset=one model per dataset (baseline); global=one shared model on all train rows",
    )
    ap.add_argument(
        "--dataset-conditioning",
        choices=["unaware", "id"],
        default="unaware",
        help="id adds dataset identity features (needed for global scope)",
    )
    args = ap.parse_args()

    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)

    print("loading snapshot...")
    snap = load_snapshot(Path(args.snapshot_manifest))
    corpus = snap.corpus
    print(
        f"  {len(corpus.environments)} envs, {len(corpus.train_rows)} train, {len(corpus.validation_rows)} val"
    )
    print(f"  scope={args.scope} dataset_conditioning={args.dataset_conditioning}")

    # metric by family
    metric_by_fam = {}
    for r in corpus.all_rows:
        metric_by_fam[r["family"]] = r["target"]["selection_metric"]

    # load summaries
    base = Path("data/meta_model/setting_to_loss_wide_v2")
    print("loading candidate summaries...")
    summaries_by_fp = {}
    for r in corpus.all_rows:
        fp = r["example_fingerprint_sha256"]
        if fp in summaries_by_fp:
            continue
        sj = load_candidate_summary(base, r)
        if sj is not None:
            summaries_by_fp[fp] = sj
    print(f"  {len(summaries_by_fp)} summaries")

    clf_names = args.classifiers.split(",")
    results = {name: {"per_env": []} for name in clf_names}

    from collections import defaultdict

    train_by_ds = defaultdict(list)
    val_by_ds = defaultdict(list)
    for r in corpus.train_rows:
        train_by_ds[r["dataset_id"]].append(r)
    for r in corpus.validation_rows:
        val_by_ds[r["dataset_id"]].append(r)

    if args.scope == "global":
        # one shared model: train on all rows, eval per-environment
        all_train = list(corpus.train_rows)
        all_val = list(corpus.validation_rows)
        fam = all_train[0]["family"]
        # metric varies by family; use a dict
        encoder = FeatureEncoder(
            feature_set=args.feature_set,
            include_parameter_count=True,
            dataset_conditioning=args.dataset_conditioning,
        )
        encoder.fit(all_train + all_val)
        print(f"  encoder: {encoder.n_output_features_} features")

        # build pairwise: use mean_loss directly (family-agnostic, lower=better)
        X_all_train = encoder.transform(all_train)
        train_raw = np.array([r["target"]["mean_loss"] for r in all_train])
        i_idx, j_idx = np.triu_indices(len(all_train), k=1)
        # subsample if too many pairs
        max_pairs = 200000
        if len(i_idx) > max_pairs:
            rng = np.random.RandomState(args.seed)
            sel = rng.choice(len(i_idx), max_pairs, replace=False)
            i_idx, j_idx = i_idx[sel], j_idx[sel]
        diffs = X_all_train[i_idx] - X_all_train[j_idx]
        labels = (train_raw[i_idx] < train_raw[j_idx]).astype(np.float64)
        X_train = np.vstack([diffs, -diffs])
        y_train = np.concatenate([labels, 1.0 - labels])
        print(f"  pairwise train: {len(X_train)} pairs (subsampled to {max_pairs})")

        X_val_all = encoder.transform(all_val)
        val_raw_all = np.array([r["target"]["mean_loss"] for r in all_val])
        val_fps_all = [r["example_fingerprint_sha256"] for r in all_val]

        # group val by environment for per-env eval
        val_by_env = defaultdict(list)
        val_env_idx = defaultdict(list)
        for i, r in enumerate(all_val):
            val_by_env[r["experiment_id"]].append(r)
            val_env_idx[r["experiment_id"]].append(i)

        for clf_name in clf_names:
            clf = make_classifier(clf_name, args.seed)
            print(f"  fitting {clf_name}...", flush=True)
            clf.fit(X_train, y_train)

            for env_id, env_rows in sorted(val_by_env.items()):
                va = env_rows
                vi = val_env_idx[env_id]
                fam = va[0]["family"]
                metric = metric_by_fam.get(fam, "test_mse")
                n_val = len(va)
                if n_val < 3:
                    continue
                X_val = X_val_all[vi]
                val_raw = val_raw_all[vi]
                val_fps = [val_fps_all[i] for i in vi]

                # filter out rows with missing summaries
                keep = [k for k, fp in enumerate(val_fps) if fp in summaries_by_fp]
                if len(keep) < 3:
                    continue
                keep = np.array(keep)
                X_val = X_val[keep]
                val_raw = val_raw[keep]
                val_fps = [val_fps[k] for k in keep]
                n_val = len(val_fps)

                idx_arr = np.array(list(combinations(range(n_val), 3)), dtype=np.int64)
                nt = idx_arr.shape[0]
                val_summaries = [summaries_by_fp[fp] for fp in val_fps]
                val_means = np.array([s[f"mean_{metric}"] for s in val_summaries])
                val_stds = np.array([s[f"std_{metric}"] for s in val_summaries])
                seed_finals = np.array(
                    [
                        [sr[f"final_{metric}"] for sr in s["seed_results"]]
                        for s in val_summaries
                    ]
                )
                t_means = val_means[idx_arr]
                t_stds = val_stds[idx_arr]
                t_seed = seed_finals[idx_arr]
                sorted_idx = np.argsort(t_means, axis=1)
                w = sorted_idx[:, 0]
                ru = sorted_idx[:, 1]
                gaps = np.abs(t_means[np.arange(nt), ru] - t_means[np.arange(nt), w])
                seed_winners = np.argmin(t_seed, axis=1)
                wins = (seed_winners == w[:, None]).sum(axis=1)
                win_rate = wins / seed_finals.shape[1]
                non_overlap = (
                    t_means[np.arange(nt), w] + t_stds[np.arange(nt), w]
                    < t_means[np.arange(nt), ru] - t_stds[np.arange(nt), ru]
                )
                sig_mask = (
                    (gaps >= args.gap_min)
                    & (win_rate >= args.win_rate_min)
                    & non_overlap
                )
                t_true_winners = np.argmin(val_raw[idx_arr], axis=1)

                votes = np.zeros((nt, 3))
                for pi, pj in [(0, 1), (0, 2), (1, 2)]:
                    diffs_t = X_val[idx_arr[:, pi]] - X_val[idx_arr[:, pj]]
                    probs = clf.predict_proba(diffs_t)[:, 1]
                    i_wins = probs >= 0.5
                    votes[i_wins, pi] += 1
                    votes[~i_wins, pj] += 1
                pred_winners = np.argmax(votes, axis=1)
                correct = pred_winners == t_true_winners

                results[clf_name]["per_env"].append(
                    {
                        "environment": env_id,
                        "dataset": va[0]["dataset_id"],
                        "family": fam,
                        "n_val_rows": n_val,
                        "n_triples": nt,
                        "n_significant": int(sig_mask.sum()),
                        "all_accuracy": float(correct.mean()),
                        "significant_accuracy": float(correct[sig_mask].mean())
                        if sig_mask.sum()
                        else None,
                    }
                )
            print(f"  {clf_name}: done", flush=True)

    else:
        # baseline: one model per dataset
        for ds_id in sorted(train_by_ds):
            tr = train_by_ds[ds_id]
            va = val_by_ds.get(ds_id, [])
            if not va or len(tr) < 10:
                continue
            fam = tr[0]["family"]
            metric = metric_by_fam.get(fam, "test_mse")

            encoder = FeatureEncoder(
                feature_set=args.feature_set,
                include_parameter_count=True,
                dataset_conditioning=args.dataset_conditioning,
            )
            encoder.fit(tr + va)

            X_train, y_train = build_pairwise_dataset(
                tr, summaries_by_fp, metric, encoder
            )
            if len(X_train) < 20:
                continue

            X_val = encoder.transform(va)
            val_fps_all_ds = [r["example_fingerprint_sha256"] for r in va]
            val_raw_all_ds = np.array([r["target"]["mean_loss"] for r in va])

            # filter out rows with missing summaries
            keep = [k for k, fp in enumerate(val_fps_all_ds) if fp in summaries_by_fp]
            if len(keep) < 3:
                continue
            keep = np.array(keep)
            X_val = X_val[keep]
            val_raw_losses = val_raw_all_ds[keep]
            val_fps = [val_fps_all_ds[k] for k in keep]
            n_val = len(val_fps)

            idx_arr = np.array(list(combinations(range(n_val), 3)), dtype=np.int64)
            nt = idx_arr.shape[0]
            val_summaries = [summaries_by_fp[fp] for fp in val_fps]
            val_means = np.array([s[f"mean_{metric}"] for s in val_summaries])
            val_stds = np.array([s[f"std_{metric}"] for s in val_summaries])
            seed_finals = np.array(
                [
                    [sr[f"final_{metric}"] for sr in s["seed_results"]]
                    for s in val_summaries
                ]
            )
            t_means = val_means[idx_arr]
            t_stds = val_stds[idx_arr]
            t_seed = seed_finals[idx_arr]
            sorted_idx = np.argsort(t_means, axis=1)
            w = sorted_idx[:, 0]
            ru = sorted_idx[:, 1]
            gaps = np.abs(t_means[np.arange(nt), ru] - t_means[np.arange(nt), w])
            seed_winners = np.argmin(t_seed, axis=1)
            wins = (seed_winners == w[:, None]).sum(axis=1)
            win_rate = wins / seed_finals.shape[1]
            non_overlap = (
                t_means[np.arange(nt), w] + t_stds[np.arange(nt), w]
                < t_means[np.arange(nt), ru] - t_stds[np.arange(nt), ru]
            )
            sig_mask = (
                (gaps >= args.gap_min) & (win_rate >= args.win_rate_min) & non_overlap
            )
            t_true_winners = np.argmin(val_raw_losses[idx_arr], axis=1)

            for clf_name in clf_names:
                clf = make_classifier(clf_name, args.seed)
                try:
                    clf.fit(X_train, y_train)
                except Exception as e:
                    print(f"  {ds_id} {clf_name} fit failed: {e}")
                    continue

                votes = np.zeros((nt, 3))
                for pi, pj in [(0, 1), (0, 2), (1, 2)]:
                    diffs = X_val[idx_arr[:, pi]] - X_val[idx_arr[:, pj]]
                    probs = clf.predict_proba(diffs)[:, 1]
                    i_wins = probs >= 0.5
                    votes[i_wins, pi] += 1
                    votes[~i_wins, pj] += 1
                pred_winners = np.argmax(votes, axis=1)
                correct = pred_winners == t_true_winners

                all_total = nt
                all_correct = int(correct.sum())
                sig_total = int(sig_mask.sum())
                sig_correct = int(correct[sig_mask].sum()) if sig_total else 0

                results[clf_name]["per_env"].append(
                    {
                        "dataset": ds_id,
                        "family": fam,
                        "n_train_rows": len(tr),
                        "n_val_rows": n_val,
                        "n_pairs_train": len(X_train),
                        "n_triples": all_total,
                        "n_significant": sig_total,
                        "all_accuracy": all_correct / all_total if all_total else None,
                        "significant_accuracy": sig_correct / sig_total
                        if sig_total
                        else None,
                    }
                )

            print(f"  {ds_id}: done ({nt} triples)", flush=True)

    # aggregate
    summary = {"classifiers": {}}
    for name, data in results.items():
        envs = data["per_env"]
        if not envs:
            continue
        mac_all = np.mean(
            [e["all_accuracy"] for e in envs if e["all_accuracy"] is not None]
        )
        mac_sig = np.mean(
            [
                e["significant_accuracy"]
                for e in envs
                if e["significant_accuracy"] is not None
            ]
        )
        tot_sig = sum(e["n_significant"] for e in envs)
        tot_all = sum(e["n_triples"] for e in envs)
        summary["classifiers"][name] = {
            "n_environments": len(envs),
            "macro_all_accuracy": float(mac_all),
            "macro_significant_accuracy": float(mac_sig),
            "total_triples": tot_all,
            "total_significant_triples": tot_sig,
            "per_environment": envs,
        }
        print(
            f"{name}: macro all={mac_all:.4f} sig={mac_sig:.4f} (sig_triples={tot_sig}/{tot_all})"
        )

    summary["config"] = {
        "feature_set": args.feature_set,
        "seed": args.seed,
        "scope": args.scope,
        "dataset_conditioning": args.dataset_conditioning,
        "significance": {
            "gap_min": args.gap_min,
            "win_rate_min": args.win_rate_min,
            "use_non_overlap": args.use_non_overlap,
        },
    }
    out_path = out / "winner_classification_results.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
