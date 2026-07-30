from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Callable
from itertools import combinations
from pathlib import Path
from typing import Any

from architecture_iq.candidates.axes import (
    choices_compatible,
    infer_axes,
    infer_question_type,
)
from architecture_iq.candidates.sets import list_candidates_in_set
from architecture_iq.paths import DATA_DIR
from architecture_iq.profile import Profile
from architecture_iq.questions.runs import (
    CANDIDATE_REUSE_POLICIES,
    DEFAULT_CANDIDATE_REUSE_POLICY,
    make_run_name,
    question_in_run_dir,
    question_run_dir,
    write_run_manifest,
)
from architecture_iq.significance.validator import load_summary, validate_significance
from architecture_iq.util import read_json, short_hash, write_json

CandidateProgress = Callable[[int, int, str], None]


def _letters(n: int) -> list[str]:
    return [chr(ord("A") + i) for i in range(n)]


def eligible_candidate_paths(paths: list[Path]) -> list[Path]:
    return [p for p in paths if not load_summary(p).get("excluded")]


def load_candidate_pool_from_sets(set_paths: list[Path]) -> list[Path]:
    pool: list[Path] = []
    for set_path in set_paths:
        pool.extend(list_candidates_in_set(set_path))
    return eligible_candidate_paths(pool)


def _cross_profile_candidate_allowed(profile: Profile, spec: dict[str, Any]) -> bool:
    source_key = (str(spec.get("profile", "")), str(spec.get("profile_hash", "")))
    if source_key == ("", "") or source_key == (profile.name, profile.profile_hash):
        return True
    model_type = str(spec.get("model", {}).get("type", ""))
    reuse = profile.raw.get("cross_profile_reuse", {})
    if reuse.get("enabled") is not True:
        return False
    allowlist = profile.raw.get("cross_profile_reuse", {}).get("transformer_allowlist", [])
    return any(
        isinstance(entry, dict)
        and entry.get("source_profile") == source_key[0]
        and entry.get("source_profile_hash") == source_key[1]
        and model_type in entry.get("model_types", [])
        for entry in allowlist
    )

def _validate_pool_dataset(pool: list[Path], dataset_path: Path) -> None:
    dataset_spec = read_json(dataset_path / "dataset_spec.json")
    dataset_id = dataset_spec["dataset_id"]
    family = dataset_spec["family"]
    for path in pool:
        spec = read_json(path / "candidate_spec.json")
        if spec["dataset_id"] != dataset_id or spec["family"] != family:
            raise ValueError(
                f"Candidate {path} belongs to {spec['family']}/{spec['dataset_id']}, "
                f"expected {family}/{dataset_id}"
            )


def find_significant_subsets(
    pool: list[Path],
    profile: Profile,
    rng: random.Random,
    *,
    num_choices: int | None = None,
    limit: int | None = None,
    max_attempts: int | None = None,
    question_type: str | None = None,
    selection_metric: str = "test_mse",
) -> list[list[Path]]:
    """Return significant candidate subsets; exhaustive when feasible."""
    num_choices = num_choices if num_choices is not None else profile.num_choices
    if len(pool) < num_choices:
        return []

    summary_map = {p: load_summary(p) for p in pool}
    n_combos = math.comb(len(pool), num_choices)
    max_exhaustive = int(
        profile.question_generation.get("max_exhaustive_combinations", 500_000)
    )

    passing: list[list[Path]] = []

    def _subset_ok(combo: tuple[Path, ...]) -> bool:
        specs = [read_json(p / "candidate_spec.json") for p in combo]
        if not choices_compatible(specs, question_type):
            return False
        sig = validate_significance(
            [summary_map[p] for p in combo],
            profile,
            metric=selection_metric,
        )
        return sig.passed

    if n_combos <= max_exhaustive:
        for combo in combinations(pool, num_choices):
            if _subset_ok(combo):
                passing.append(list(combo))
    else:
        attempts = max_attempts if max_attempts is not None else int(
            profile.question_generation["max_attempts"]
        )
        seen: set[frozenset[str]] = set()
        for _ in range(attempts):
            paths = rng.sample(pool, num_choices)
            key = frozenset(p.name for p in paths)
            if key in seen:
                continue
            seen.add(key)
            if _subset_ok(tuple(paths)):
                passing.append(paths)

    if not passing:
        return []

    rng.shuffle(passing)
    if limit is not None:
        return passing[:limit]
    return passing


def select_significant_candidates(
    pool: list[Path],
    profile: Profile,
    rng: random.Random,
    *,
    num_choices: int | None = None,
    max_attempts: int | None = None,
) -> list[Path] | None:
    subsets = find_significant_subsets(
        pool,
        profile,
        rng,
        num_choices=num_choices,
        limit=1,
        max_attempts=max_attempts,
    )
    return subsets[0] if subsets else None


def _candidate_set_key(paths: list[Path]) -> frozenset[str]:
    return frozenset(p.name for p in paths)


def _unique_subsets(subsets: list[list[Path]]) -> list[list[Path]]:
    seen: set[frozenset[str]] = set()
    unique: list[list[Path]] = []
    for subset in subsets:
        key = _candidate_set_key(subset)
        if key not in seen:
            seen.add(key)
            unique.append(subset)
    return unique


def _pick_unique_subsets(subsets: list[list[Path]], num_questions: int) -> list[list[Path]]:
    """Pick distinct candidate sets while allowing candidates across questions to recur."""
    return _unique_subsets(subsets)[:num_questions]


def _pick_balanced_unique_pairs(
    subsets: list[list[Path]],
    num_questions: int,
    *,
    profile: Profile,
    selection_metric: str,
    model_types: dict[Path, str],
    max_winner_fraction: float,
) -> list[list[Path]]:
    """Pick unique cross-type pairs while capping each winner type.

    ``subsets`` is already shuffled by the caller's RNG.  We preserve that
    order within winner buckets, so selection is deterministic for a fixed RNG
    state while still filling both winner types before one can exceed the cap.
    """
    if num_questions <= 0:
        return []
    if not 0 < float(max_winner_fraction) <= 1:
        raise ValueError("max_winner_fraction must be in (0, 1]")
    unique = _unique_subsets(subsets)
    buckets: dict[str, list[list[Path]]] = {}
    for subset in unique:
        if len(subset) != 2:
            continue
        sig = validate_significance(
            [load_summary(path) for path in subset],
            profile,
            metric=selection_metric,
        )
        if not sig.passed or sig.winner_index < 0:
            continue
        winner_type = model_types[subset[sig.winner_index]]
        buckets.setdefault(winner_type, []).append(subset)
    if not buckets:
        return []
    cap = int(math.floor(num_questions * float(max_winner_fraction)))
    cap = max(1, cap)
    if sum(len(items) for items in buckets.values()) < num_questions:
        return []
    selected: list[list[Path]] = []
    counts: Counter[str] = Counter()
    # Always choose from the currently least-used eligible winner type. This
    # gives a balanced run when both types are available and enforces the cap.
    while len(selected) < num_questions:
        choices = [
            winner_type
            for winner_type, items in buckets.items()
            if items and counts[winner_type] < cap
        ]
        if not choices:
            return []
        winner_type = min(choices, key=lambda item: (counts[item], item))
        selected.append(buckets[winner_type].pop(0))
        counts[winner_type] += 1
    return selected

def _pick_candidate_disjoint_subsets(
    subsets: list[list[Path]],
    num_questions: int,
) -> list[list[Path]]:
    """Pick candidate-disjoint subsets deterministically.

    Small pools retain the exact search used historically. Large pools first
    use the current subset order greedily, then spend a bounded node budget on
    backtracking so pathological subset counts cannot cause an unbounded
    recursive search.
    """
    seen: set[frozenset[str]] = set()
    unique: list[tuple[list[Path], frozenset[str]]] = []
    for subset in subsets:
        key = _candidate_set_key(subset)
        if key in seen:
            continue
        seen.add(key)
        unique.append((subset, key))

    if num_questions <= 0 or not unique:
        return []

    def search(
        start: int,
        used_candidate_ids: frozenset[str],
        picked: list[list[Path]],
    ) -> list[list[Path]] | None:
        if len(picked) == num_questions:
            return picked
        if len(unique) - start < num_questions - len(picked):
            return None
        for index in range(start, len(unique)):
            subset, candidate_ids = unique[index]
            if not candidate_ids.isdisjoint(used_candidate_ids):
                continue
            result = search(
                index + 1,
                used_candidate_ids | candidate_ids,
                [*picked, subset],
            )
            if result is not None:
                return result
        return None

    exact_limit = 200
    if len(unique) <= exact_limit:
        return search(0, frozenset(), []) or []

    picked: list[list[Path]] = []
    used_candidate_ids: frozenset[str] = frozenset()
    for subset, candidate_ids in unique:
        if candidate_ids.isdisjoint(used_candidate_ids):
            picked.append(subset)
            used_candidate_ids = used_candidate_ids | candidate_ids
            if len(picked) == num_questions:
                return picked

    node_budget = 20_000
    nodes = 0

    def bounded_search(
        start: int,
        used: frozenset[str],
        selected: list[list[Path]],
    ) -> list[list[Path]] | None:
        nonlocal nodes
        nodes += 1
        if nodes > node_budget:
            return None
        if len(selected) == num_questions:
            return selected
        if len(unique) - start < num_questions - len(selected):
            return None
        for index in range(start, len(unique)):
            subset, candidate_ids = unique[index]
            if not candidate_ids.isdisjoint(used):
                continue
            result = bounded_search(
                index + 1,
                used | candidate_ids,
                [*selected, subset],
            )
            if result is not None:
                return result
            if nodes > node_budget:
                return None
        return None

    return bounded_search(0, frozenset(), []) or []


def _pick_bounded_reuse_pairs(
    subsets: list[list[Path]],
    num_questions: int,
    *,
    max_candidate_uses: int,
    model_types: dict[Path, str],
    required_model_types: frozenset[str],
) -> list[list[Path]]:
    """Pick distinct two-choice pairs subject to a per-candidate use cap.

    The declared model types make the graph bipartite. A max-flow selection
    avoids a greedy ordering falsely reporting that an available pool is full.
    """
    if num_questions <= 0 or max_candidate_uses < 1:
        return []
    if len(required_model_types) != 2:
        raise ValueError("Bounded reuse requires exactly two model types")

    left_type, right_type = sorted(required_model_types)
    pairs: list[tuple[Path, Path]] = []
    for subset in _unique_subsets(subsets):
        if len(subset) != 2:
            continue
        first, second = subset
        if model_types[first] == left_type and model_types[second] == right_type:
            pairs.append((first, second))
        elif model_types[first] == right_type and model_types[second] == left_type:
            pairs.append((second, first))

    left_nodes = sorted({left for left, _ in pairs}, key=lambda path: path.name)
    right_nodes = sorted({right for _, right in pairs}, key=lambda path: path.name)
    source = 0
    left_start = 1
    right_start = left_start + len(left_nodes)
    sink = right_start + len(right_nodes)
    graph: list[list[list[int]]] = [[] for _ in range(sink + 1)]

    def add_edge(start: int, end: int, capacity: int) -> int:
        edge_index = len(graph[start])
        graph[start].append([end, capacity, len(graph[end])])
        graph[end].append([start, 0, edge_index])
        return edge_index

    left_index = {path: left_start + index for index, path in enumerate(left_nodes)}
    right_index = {path: right_start + index for index, path in enumerate(right_nodes)}
    for path in left_nodes:
        add_edge(source, left_index[path], max_candidate_uses)
    for path in right_nodes:
        add_edge(right_index[path], sink, max_candidate_uses)

    pair_edges: list[tuple[list[Path], int, int]] = []
    for left, right in pairs:
        edge_index = add_edge(left_index[left], right_index[right], 1)
        pair_edges.append(([left, right], left_index[left], edge_index))

    total_flow = 0
    while True:
        level = [-1] * len(graph)
        level[source] = 0
        queue = [source]
        for node in queue:
            for end, capacity, _ in graph[node]:
                if capacity and level[end] < 0:
                    level[end] = level[node] + 1
                    queue.append(end)
        if level[sink] < 0:
            break

        next_edge = [0] * len(graph)

        def send(node: int, flow: int) -> int:
            if node == sink:
                return flow
            while next_edge[node] < len(graph[node]):
                edge_index = next_edge[node]
                end, capacity, reverse_index = graph[node][edge_index]
                if capacity and level[end] == level[node] + 1:
                    pushed = send(end, min(flow, capacity))
                    if pushed:
                        graph[node][edge_index][1] -= pushed
                        graph[end][reverse_index][1] += pushed
                        return pushed
                next_edge[node] += 1
            return 0

        while pushed := send(source, 10**9):
            total_flow += pushed

    if total_flow < num_questions:
        return []
    selected = [
        pair
        for pair, left_node, edge_index in pair_edges
        if graph[left_node][edge_index][1] == 0
    ]
    return selected[:num_questions]


def _budget_field(specs: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {spec["budget"]["total_samples_seen"] for spec in specs}
    if len(totals) == 1:
        return {"total_samples_seen": next(iter(totals))}
    return {"total_samples_seen": sorted(totals), "mixed": True}


def build_question_record(
    profile: Profile,
    *,
    dataset_spec: dict[str, Any],
    dataset_path: Path,
    candidate_paths: list[Path],
    candidate_set_paths: list[Path],
    rng: random.Random,
) -> dict[str, Any]:
    summaries = [load_summary(p) for p in candidate_paths]
    specs = [read_json(p / "candidate_spec.json") for p in candidate_paths]
    if not choices_compatible(specs):
        invariant, varying = infer_axes(specs)
        raise ValueError(
            "Candidates are not compatible for a question "
            f"(invariant={invariant}, varying={varying})"
        )

    question_type = infer_question_type(specs)
    invariant_axes, varying_axes = infer_axes(specs)
    execution_device = str(specs[0].get("execution", {}).get("device", "cpu"))

    sig = validate_significance(summaries, profile, metric=dataset_spec["selection_metric"])
    if not sig.passed:
        raise ValueError(f"Significance failed: {sig.reason}")

    for path in candidate_paths:
        if not _cross_profile_candidate_allowed(profile, read_json(path / "candidate_spec.json")):
            raise ValueError(f"Candidate {path.name} is not allowed by profile cross_profile_reuse allowlist")
        cand_budget = read_json(path / "candidate_spec.json")["budget"]
        steps = cand_budget["training_steps"]
        bs = cand_budget["batch_size"]
        total = cand_budget["total_samples_seen"]
        if steps * bs != total:
            raise ValueError(
                f"Candidate {path.name} violates batch_size × training_steps = "
                f"total_samples_seen ({bs} × {steps} != {total})"
            )

    winner_path = candidate_paths[sig.winner_index]
    others = [p for i, p in enumerate(candidate_paths) if i != sig.winner_index]
    rng.shuffle(others)
    ordered_paths = [winner_path] + others
    rng.shuffle(ordered_paths)

    letters = _letters(len(ordered_paths))
    choices = []
    correct_letter = "A"
    data_root = DATA_DIR.resolve()
    for letter, path in zip(letters, ordered_paths):
        path = path.resolve()
        spec = read_json(path / "candidate_spec.json")
        choices.append(
            {
                "letter": letter,
                "candidate_id": spec["candidate_id"],
                "candidate_path": str(path.relative_to(data_root)),
                "candidate_set_path": str(path.parent.relative_to(data_root)),
            }
        )
        if path == winner_path:
            correct_letter = letter

    body = {
        "schema_version": profile.schema_version,
        "profile_hash": profile.profile_hash,
        "family": dataset_spec["family"],
        "dataset_id": dataset_spec["dataset_id"],
        "budget": _budget_field(specs),
        "type": question_type,
        "invariant_axes": invariant_axes,
        "varying_axes": varying_axes,
        "candidate_sets": [
            str(p.resolve().relative_to(data_root)) for p in candidate_set_paths
        ],
        "num_choices": len(choices),
        "choices": choices,
        "correct_letter": correct_letter,
        "significance": {
            "passed": sig.passed,
            "gap": sig.gap,
            "win_rate": sig.win_rate,
            "metric": sig.metric,
        },
        "evaluation": {
            "selection_metric": dataset_spec["selection_metric"],
            "n_seeds": profile.n_seeds,
            "base_seed": profile.base_seed,
            "device": execution_device,
        },
        "prompt": {
            "template_version": profile.prompts["template_version"],
            "rendered_path": "prompt.txt",
        },
    }
    qid = f"q_{short_hash(body)}"
    body["question_id"] = qid
    body["profile"] = profile.name
    return body


def _write_question(
    profile: Profile,
    *,
    dataset_spec: dict[str, Any],
    dataset_path: Path,
    candidate_paths: list[Path],
    candidate_set_paths: list[Path],
    run_path: Path,
    run_name: str,
    rng: random.Random,
) -> tuple[dict[str, Any], Path]:
    record = build_question_record(
        profile,
        dataset_spec=dataset_spec,
        dataset_path=dataset_path,
        candidate_paths=candidate_paths,
        candidate_set_paths=candidate_set_paths,
        rng=rng,
    )
    data_root = DATA_DIR.resolve()
    record["question_run_id"] = run_name
    record["question_run_path"] = str(run_path.resolve().relative_to(data_root))

    out = question_in_run_dir(run_path, record["question_id"])
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "question.json", record)
    return record, out


def generate_questions(
    profile: Profile,
    *,
    dataset_path: Path,
    candidate_set_paths: list[Path],
    rng: random.Random,
    num_questions: int = 1,
    num_choices: int | None = None,
    seed: int = 0,
    require_distinct_model_types: bool = False,
    required_model_types: frozenset[str] | None = None,
    candidate_reuse_policy: str = DEFAULT_CANDIDATE_REUSE_POLICY,
    max_candidate_uses: int | None = None,
    winner_type_max_fraction: float | None = None,
) -> tuple[Path, list[tuple[dict[str, Any], Path]]]:
    if num_questions < 1:
        raise ValueError("num_questions must be at least 1")
    if not candidate_set_paths:
        raise ValueError("At least one candidate set path is required")
    if candidate_reuse_policy not in CANDIDATE_REUSE_POLICIES:
        raise ValueError(f"Unknown candidate reuse policy: {candidate_reuse_policy}")

    resolved_sets = [p.resolve() for p in candidate_set_paths]
    pool = load_candidate_pool_from_sets(resolved_sets)
    pool = [path for path in pool if _cross_profile_candidate_allowed(profile, read_json(path / "candidate_spec.json"))]
    _validate_pool_dataset(pool, dataset_path)

    dataset_spec = read_json(dataset_path / "dataset_spec.json")
    n_choices = num_choices if num_choices is not None else profile.num_choices
    if len(pool) < n_choices:
        raise RuntimeError(
            f"Not enough eligible candidates ({len(pool)}) for {n_choices} choices"
        )
    if required_model_types is not None and len(required_model_types) != n_choices:
        raise ValueError("required_model_types must contain one distinct type per choice")
    if candidate_reuse_policy != DEFAULT_CANDIDATE_REUSE_POLICY:
        if n_choices != 2 or not required_model_types:
            raise ValueError("Candidate-reuse policies require two explicit model types")
        if candidate_reuse_policy == "sequential_bounded_reuse":
            if not isinstance(max_candidate_uses, int) or isinstance(max_candidate_uses, bool) or max_candidate_uses < 1:
                raise ValueError("sequential_bounded_reuse requires max_candidate_uses >= 1")
        elif max_candidate_uses is not None:
            raise ValueError("blind_pair_unique does not use max_candidate_uses")

    subsets = find_significant_subsets(
        pool,
        profile,
        rng,
        num_choices=n_choices,
        selection_metric=dataset_spec["selection_metric"],
    )
    model_types = {
        candidate_path: read_json(candidate_path / "candidate_spec.json")["model"]["type"]
        for candidate_path in pool
    }
    if required_model_types is not None:
        subsets = [
            subset
            for subset in subsets
            if {model_types[candidate_path] for candidate_path in subset}
            == required_model_types
        ]
    elif require_distinct_model_types:
        subsets = [
            subset
            for subset in subsets
            if len({model_types[candidate_path] for candidate_path in subset}) == n_choices
        ]
    if not subsets:
        raise RuntimeError(
            f"Failed to find significant {n_choices}-candidate subsets in pool of {len(pool)}"
        )

    if candidate_reuse_policy == DEFAULT_CANDIDATE_REUSE_POLICY:
        selected_sets = _pick_candidate_disjoint_subsets(subsets, num_questions)
    elif candidate_reuse_policy == "blind_pair_unique":
        if winner_type_max_fraction is None:
            selected_sets = _pick_unique_subsets(subsets, num_questions)
        else:
            selected_sets = _pick_balanced_unique_pairs(
                subsets,
                num_questions,
                profile=profile,
                selection_metric=dataset_spec["selection_metric"],
                model_types=model_types,
                max_winner_fraction=winner_type_max_fraction,
            )
    else:
        selected_sets = _pick_bounded_reuse_pairs(
            subsets,
            num_questions,
            max_candidate_uses=max_candidate_uses or 0,
            model_types=model_types,
            required_model_types=required_model_types or frozenset(),
        )
    if len(selected_sets) < num_questions:
        requirement = (
            "candidate-disjoint"
            if candidate_reuse_policy == DEFAULT_CANDIDATE_REUSE_POLICY
            else candidate_reuse_policy
        )
        raise RuntimeError(
            f"Requested {num_questions} {requirement} questions but no valid "
            f"selection exists among {len(subsets)} significant subsets. Generate "
            "more candidates or request fewer questions."
        )

    dataset_spec = read_json(dataset_path / "dataset_spec.json")
    dataset_id = dataset_spec["dataset_id"]
    family = dataset_spec["family"]

    run_name = make_run_name(
        num_questions=num_questions,
        num_choices=n_choices,
        candidate_set_names=[p.name for p in resolved_sets],
        salt=rng.randint(0, 2**31 - 1),
    )
    run_path = question_run_dir(dataset_path.resolve(), run_name)
    run_path.mkdir(parents=True, exist_ok=False)

    results: list[tuple[dict[str, Any], Path]] = []
    for selected in selected_sets:
        results.append(
            _write_question(
                profile,
                dataset_spec=dataset_spec,
                dataset_path=dataset_path,
                candidate_paths=selected,
                candidate_set_paths=resolved_sets,
                run_path=run_path,
                run_name=run_name,
                rng=rng,
            )
        )

    selected_profile_provenance: dict[tuple[str, str], dict[str, Any]] = {}
    for selected in selected_sets:
        for candidate_path in selected:
            spec = read_json(candidate_path / "candidate_spec.json")
            key = (str(spec.get("profile", "")), str(spec.get("profile_hash", "")))
            entry = selected_profile_provenance.setdefault(
                key,
                {"profile": key[0], "profile_hash": key[1], "candidate_ids": [], "model_types": []},
            )
            entry["candidate_ids"].append(str(spec["candidate_id"]))
            model_type = str(spec.get("model", {}).get("type", ""))
            if model_type not in entry["model_types"]:
                entry["model_types"].append(model_type)
    candidate_profile_provenance = sorted(
        selected_profile_provenance.values(),
        key=lambda item: (item["profile"], item["profile_hash"]),
    )
    for entry in candidate_profile_provenance:
        entry["candidate_ids"] = sorted(set(entry["candidate_ids"]))
        entry["model_types"] = sorted(entry["model_types"])

    write_run_manifest(
        run_path,
        run_name=run_name,
        profile=profile,
        dataset_id=dataset_id,
        family=family,
        candidate_set_paths=resolved_sets,
        num_questions=num_questions,
        num_choices=n_choices,
        seed=seed,
        question_ids=[record["question_id"] for record, _ in results],
        candidate_reuse_policy=candidate_reuse_policy,
        run_purpose=(
            "review_blind_pool"
            if candidate_reuse_policy == "blind_pair_unique"
            else "review_practice_pool"
            if candidate_reuse_policy == "sequential_bounded_reuse"
            else None
        ),
        canonical_blind_evaluation=(
            False if candidate_reuse_policy != DEFAULT_CANDIDATE_REUSE_POLICY else None
        ),
        max_candidate_uses=max_candidate_uses,
        pair_reuse_policy=(
            "unique" if candidate_reuse_policy != DEFAULT_CANDIDATE_REUSE_POLICY else None
        ),
        required_model_types=(
            sorted(required_model_types) if required_model_types is not None else None
        ),
        winner_type_max_fraction=winner_type_max_fraction,
        candidate_profile_provenance=candidate_profile_provenance,
    )
    return run_path, results
