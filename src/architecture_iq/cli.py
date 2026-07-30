from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import typer

from architecture_iq.candidates.sets import (
    generate_candidate_set,
    parse_model_type_counts,
    parse_varying_axes,
)
from architecture_iq.datasets import (
    create_dataset,
    format_dataset_summary_lines,
    load_dataset_spec,
    resolve_dataset_family,
)
from architecture_iq.interactive import (
    interactive_create_dataset,
    interactive_generate_candidate_set,
    interactive_generate_questions,
)
from architecture_iq.profile import load_profile
from architecture_iq.prompts.renderer import write_prompt
from architecture_iq.questions.generator import generate_questions
from architecture_iq.registry import ensure_registries

app = typer.Typer(help="ArchitectureIQ benchmark CLI")
ensure_registries()


def _reject_interactive_flags(interactive: bool, **flags: bool) -> None:
    if not interactive:
        return
    bad = [name.replace("_", "-") for name, set_flag in flags.items() if set_flag]
    if bad:
        raise typer.BadParameter(
            "Interactive mode does not accept other arguments; use only -i/--interactive "
            f"(got: {', '.join('--' + name for name in bad)})"
        )


@app.command("create-dataset")
def create_dataset_cmd(
    profile: str = typer.Option("v1", help="Profile name"),
    family: Optional[str] = typer.Option(
        None,
        help="Dataset family from profile pool",
    ),
    random_family: bool = typer.Option(
        False,
        "--random-family",
        help="Pick a random family from the profile pool",
    ),
    seed: Optional[int] = typer.Option(
        None,
        help="Instance seed (default 0 when omitted)",
    ),
    input_dim: Optional[int] = typer.Option(
        None,
        "--input-dim",
        help="Input dimension for multivariate_regression or synthetic_tabular_classification "
        "(must be in that family's profile input_dims pool)",
    ),
    rule_family: Optional[str] = typer.Option(
        None,
        "--rule-family",
        help="Rule family for synthetic_tabular_classification "
        "(e.g. xor, spiral; must be in profile rule_families)",
    ),
    interactive: bool = typer.Option(
        False,
        "--interactive",
        "-i",
        help="Prompt for family and seed, then create a new dataset",
    ),
) -> None:
    """Create a new dataset instance."""
    prof = load_profile(profile)
    rng = random.Random()

    _reject_interactive_flags(
        interactive,
        family=family is not None,
        random_family=random_family,
        seed=seed is not None,
        input_dim=input_dim is not None,
        rule_family=rule_family is not None,
    )

    if interactive:
        spec, path = interactive_create_dataset(
            prof,
            rng=rng,
            write=typer.echo,
        )
        for line in format_dataset_summary_lines(spec):
            typer.echo(line)
        typer.echo(f"Path: {path}")
        return

    if family is None and not random_family:
        raise typer.BadParameter(
            "Specify --family, --random-family, or --interactive"
        )
    if family is not None and random_family:
        raise typer.BadParameter("Use only one of --family and --random-family")
    _dim_families = ("multivariate_regression", "synthetic_tabular_classification")
    if input_dim is not None and family not in (None, *_dim_families):
        raise typer.BadParameter(
            "--input-dim is only valid with multivariate_regression or "
            "synthetic_tabular_classification"
        )
    if rule_family is not None and family not in (None, "synthetic_tabular_classification"):
        raise typer.BadParameter(
            "--rule-family is only valid with --family synthetic_tabular_classification"
        )

    instance_seed = seed if seed is not None else 0
    family_name = resolve_dataset_family(
        prof,
        family=family,
        random_pick=random_family,
        rng=rng,
    )
    if input_dim is not None and family_name not in _dim_families:
        raise typer.BadParameter(
            "--input-dim requires multivariate_regression or "
            f"synthetic_tabular_classification (got random family {family_name!r})"
        )
    if rule_family is not None and family_name != "synthetic_tabular_classification":
        raise typer.BadParameter(
            "--rule-family requires synthetic_tabular_classification "
            f"(got {family_name!r})"
        )

    family_options: dict = {}
    if input_dim is not None:
        family_options["input_dim"] = input_dim
    if rule_family is not None:
        family_options["rule_family"] = rule_family
    try:
        spec, path = create_dataset(
            prof,
            instance_seed,
            family_name=family_name,
            family_options=family_options or None,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Created dataset {spec['dataset_id']} at {path}")
    for line in format_dataset_summary_lines(spec):
        typer.echo(line)


@app.command("generate-candidates")
def generate_candidates_cmd(
    dataset_path: Optional[Path] = typer.Argument(
        None,
        help="Path to dataset instance dir (required unless --interactive)",
    ),
    budget: Optional[int] = typer.Option(
        None,
        help="total_samples_seen (uses the family default when configured)",
    ),
    count: Optional[int] = typer.Option(None, help="Number of candidates to generate"),
    vary: list[str] = typer.Option(
        [],
        "--vary",
        help="Axis that may vary: model, optimizer, loss (repeat flag)",
    ),
    model_type_count: list[str] = typer.Option(
        [],
        "--model-type-count",
        help="Exact model quota as model_type=count; repeat for each type (requires --vary model)",
    ),
    profile: str = typer.Option("v1"),
    device: Optional[str] = typer.Option(
        None,
        "--device",
        help="Execution device for newly generated candidates: cpu or cuda",
    ),
    seed: int = typer.Option(0),
    interactive: bool = typer.Option(
        False,
        "--interactive",
        "-i",
        help="Prompt for varying/invariant axes and fixed component values",
    ),
) -> None:
    """Generate a named candidate set with ground truth."""
    prof = load_profile(profile)
    rng = random.Random(seed)
    model_type_counts: dict[str, int] | None = None

    _reject_interactive_flags(
        interactive,
        dataset_path=dataset_path is not None,
        budget=budget is not None,
        count=count is not None,
        vary=bool(vary),
        model_type_count=bool(model_type_count),
        device=device is not None,
        seed=seed != 0,
    )

    if interactive:
        params = interactive_generate_candidate_set(
            prof,
            rng=rng,
            write=typer.echo,
        )
        dataset_path = params["dataset_path"]
        budget = params["budget"]
        count = params["count"]
        varying_axes = params["varying_axes"]
        fixed_shared = params["fixed_shared"]
        seed = params["seed"]
        rng = random.Random(seed)
    else:
        if dataset_path is None:
            raise typer.BadParameter("dataset_path is required unless --interactive is set")
        if budget is None:
            family = load_dataset_spec(dataset_path)["family"]
            defaults = prof.family_training_defaults(family)
            if not defaults:
                raise typer.BadParameter(
                    "--budget is required unless the dataset family has a training default"
                )
            budget = defaults["total_samples_seen"]
        if count is None:
            raise typer.BadParameter("--count is required unless --interactive is set")
        if not vary:
            raise typer.BadParameter("At least one --vary axis is required unless --interactive")
        varying_axes = parse_varying_axes(vary)
        fixed_shared = None
        if model_type_count:
            try:
                model_type_counts = parse_model_type_counts(model_type_count)
            except ValueError as exc:
                raise typer.BadParameter(str(exc)) from exc
            if "model" not in varying_axes:
                raise typer.BadParameter("--model-type-count requires --vary model")

    assert dataset_path is not None and budget is not None and count is not None
    set_path = generate_candidate_set(
        prof,
        dataset_path=dataset_path,
        budget=budget,
        count=count,
        varying_axes=varying_axes,
        rng=rng,
        fixed_shared=fixed_shared,
        model_type_counts=model_type_counts,
        seed=seed,
        on_progress=lambda i, total, cid: typer.echo(f"[{i}/{total}] {cid}"),
        execution_device=device,
    )
    typer.echo(f"Candidate set written to {set_path}")


@app.command("generate-question")
def generate_question_cmd(
    dataset_path: Optional[Path] = typer.Argument(
        None,
        help="Path to dataset instance dir (required unless --interactive)",
    ),
    candidate_sets: list[Path] = typer.Argument(
        default=[],
        help="Candidate set dirs under dataset/candidates/",
    ),
    num_questions: Optional[int] = typer.Option(
        None,
        help="Number of questions to generate",
    ),
    num_choices: Optional[int] = typer.Option(
        None,
        help="Number of choices (default: profile num_choices)",
    ),
    profile: str = typer.Option("v1"),
    seed: int = typer.Option(0),
    candidate_reuse_policy: str = typer.Option(
        "globally_disjoint_within_run",
        "--candidate-reuse-policy",
        help="Run-level candidate reuse policy",
    ),
    max_candidate_uses: Optional[int] = typer.Option(
        None,
        "--max-candidate-uses",
        help="Required only for sequential_bounded_reuse",
    ),
    winner_type_max_fraction: Optional[float] = typer.Option(
        None,
        "--winner-type-max-fraction",
        help="Maximum fraction of questions won by any one model type (for example 0.70)",
    ),
    required_model_types: list[str] = typer.Option(
        [],
        "--required-model-type",
        help="Repeat once per required model type (for example: gru_lm and transformer_lm)",
    ),
    gap_max: Optional[float] = typer.Option(
        None,
        "--gap-max",
        help="Optional: reject subsets whose winner–runner-up metric gap exceeds this "
        "(overrides profile question_generation.quality.gap_max when passed)",
    ),
    gap_worst_max: Optional[float] = typer.Option(
        None,
        "--gap-worst-max",
        help="Optional: reject subsets whose winner–worst metric gap exceeds this "
        "(overrides profile question_generation.quality.gap_worst_max when passed)",
    ),
    require_finite_mean: Optional[bool] = typer.Option(
        None,
        "--require-finite-mean/--allow-nonfinite-mean",
        help="Optional pool wash: drop candidates with non-finite selection mean",
    ),
    max_failed_seeds: Optional[int] = typer.Option(
        None,
        "--max-failed-seeds",
        help="Optional pool wash: drop candidates with more failed seeds than N "
        "(overrides profile; use 0 for no failed seeds)",
    ),
    question_type: Optional[str] = typer.Option(
        None,
        "--question-type",
        help="Optional target question type: architecture_only, optimizer_only, "
        "loss_only, or mixed. When set, subsets must match it.",
    ),
    interactive: bool = typer.Option(
        False,
        "--interactive",
        "-i",
        help="Prompt for dataset, candidate sets, and question parameters",
    ),
) -> None:
    """Assemble questions from one or more candidate sets."""
    from architecture_iq.questions.quality import QuestionQualityFilters

    prof = load_profile(profile)

    _reject_interactive_flags(
        interactive,
        dataset_path=dataset_path is not None,
        candidate_sets=bool(candidate_sets),
        num_questions=num_questions is not None,
        num_choices=num_choices is not None,
        seed=seed != 0,
        candidate_reuse_policy=candidate_reuse_policy != "globally_disjoint_within_run",
        max_candidate_uses=max_candidate_uses is not None,
        winner_type_max_fraction=winner_type_max_fraction is not None,
        required_model_types=bool(required_model_types),
        gap_max=gap_max is not None,
        gap_worst_max=gap_worst_max is not None,
        require_finite_mean=require_finite_mean is not None,
        max_failed_seeds=max_failed_seeds is not None,
        question_type=question_type is not None,
    )

    if interactive:
        params = interactive_generate_questions(prof, rng=random.Random(), write=typer.echo)
        dataset_path = params["dataset_path"]
        candidate_sets = params["candidate_set_paths"]
        num_questions = params["num_questions"]
        num_choices = params["num_choices"]
        seed = params["seed"]

    if dataset_path is None:
        raise typer.BadParameter("dataset_path is required unless --interactive is set")
    if num_questions is None:
        raise typer.BadParameter("--num-questions is required unless --interactive is set")
    if not candidate_sets:
        raise typer.BadParameter("At least one candidate set path is required")

    n_choices = num_choices if num_choices is not None else prof.num_choices
    if n_choices < 2:
        raise typer.BadParameter("num_choices must be at least 2")
    if num_questions < 1:
        raise typer.BadParameter("num_questions must be at least 1")
    if gap_max is not None and gap_max < 0:
        raise typer.BadParameter("--gap-max must be non-negative")
    if gap_worst_max is not None and gap_worst_max < 0:
        raise typer.BadParameter("--gap-worst-max must be non-negative")
    if max_failed_seeds is not None and max_failed_seeds < 0:
        raise typer.BadParameter("--max-failed-seeds must be non-negative")
    if question_type is not None and question_type not in (
        "architecture_only", "optimizer_only", "loss_only", "mixed"
    ):
        raise typer.BadParameter("--question-type must be one of architecture_only, optimizer_only, loss_only, mixed")

    for set_path in candidate_sets:
        if not set_path.is_dir():
            raise typer.BadParameter(f"Candidate set not found: {set_path}")

    quality = QuestionQualityFilters.from_profile(prof).overlay(
        gap_max=gap_max,
        gap_worst_max=gap_worst_max,
        require_finite_mean=require_finite_mean,
        max_failed_seeds=max_failed_seeds,
        gap_max_provided=gap_max is not None,
        gap_worst_max_provided=gap_worst_max is not None,
        max_failed_seeds_provided=max_failed_seeds is not None,
    )

    rng = random.Random(seed)
    run_path, results = generate_questions(
        prof,
        dataset_path=dataset_path,
        candidate_set_paths=candidate_sets,
        rng=rng,
        num_questions=num_questions,
        num_choices=n_choices,
        seed=seed,
        required_model_types=frozenset(required_model_types) or None,
        candidate_reuse_policy=candidate_reuse_policy,
        max_candidate_uses=max_candidate_uses,
        winner_type_max_fraction=winner_type_max_fraction,
        quality=quality,
        question_type=question_type,
    )

    typer.echo(f"Question run written to {run_path}")
    if quality.any_enabled:
        typer.echo(f"Quality filters: {quality.as_dict()}")
    for record, out in results:
        write_prompt(out)
        typer.echo(
            f"Question {record['question_id']} type={record['type']} "
            f"varying={record['varying_axes']} correct={record['correct_letter']} at {out}"
        )



if __name__ == "__main__":
    app()
