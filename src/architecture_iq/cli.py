from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import typer

from architecture_iq.candidates.sets import (
    generate_candidate_set,
    parse_varying_axes,
)
from architecture_iq.datasets import create_dataset, resolve_dataset_family
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
    )

    if interactive:
        spec, path = interactive_create_dataset(
            prof,
            rng=rng,
            write=typer.echo,
        )
        typer.echo(f"Expression: {spec['params']['expression']}")
        typer.echo(f"Path: {path}")
        return

    if family is None and not random_family:
        raise typer.BadParameter(
            "Specify --family, --random-family, or --interactive"
        )
    if family is not None and random_family:
        raise typer.BadParameter("Use only one of --family and --random-family")

    instance_seed = seed if seed is not None else 0
    family_name = resolve_dataset_family(
        prof,
        family=family,
        random_pick=random_family,
        rng=rng,
    )
    spec, path = create_dataset(prof, instance_seed, family_name=family_name)
    typer.echo(f"Created dataset {spec['dataset_id']} at {path}")
    typer.echo(f"Expression: {spec['params']['expression']}")


@app.command("generate-candidates")
def generate_candidates_cmd(
    dataset_path: Optional[Path] = typer.Argument(
        None,
        help="Path to dataset instance dir (required unless --interactive)",
    ),
    budget: Optional[int] = typer.Option(None, help="total_samples_seen"),
    count: Optional[int] = typer.Option(None, help="Number of candidates to generate"),
    vary: list[str] = typer.Option(
        [],
        "--vary",
        help="Axis that may vary: model, optimizer, loss (repeat flag)",
    ),
    profile: str = typer.Option("v1"),
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

    _reject_interactive_flags(
        interactive,
        dataset_path=dataset_path is not None,
        budget=budget is not None,
        count=count is not None,
        vary=bool(vary),
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
            raise typer.BadParameter("--budget is required unless --interactive is set")
        if count is None:
            raise typer.BadParameter("--count is required unless --interactive is set")
        if not vary:
            raise typer.BadParameter("At least one --vary axis is required unless --interactive")
        varying_axes = parse_varying_axes(vary)
        fixed_shared = None

    assert dataset_path is not None and budget is not None and count is not None
    set_path = generate_candidate_set(
        prof,
        dataset_path=dataset_path,
        budget=budget,
        count=count,
        varying_axes=varying_axes,
        rng=rng,
        fixed_shared=fixed_shared,
        seed=seed,
        on_progress=lambda i, total, cid: typer.echo(f"[{i}/{total}] {cid}"),
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
    interactive: bool = typer.Option(
        False,
        "--interactive",
        "-i",
        help="Prompt for dataset, candidate sets, and question parameters",
    ),
) -> None:
    """Assemble questions from one or more candidate sets."""
    prof = load_profile(profile)

    _reject_interactive_flags(
        interactive,
        dataset_path=dataset_path is not None,
        candidate_sets=bool(candidate_sets),
        num_questions=num_questions is not None,
        num_choices=num_choices is not None,
        seed=seed != 0,
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

    for set_path in candidate_sets:
        if not set_path.is_dir():
            raise typer.BadParameter(f"Candidate set not found: {set_path}")

    rng = random.Random(seed)
    run_path, results = generate_questions(
        prof,
        dataset_path=dataset_path,
        candidate_set_paths=candidate_sets,
        rng=rng,
        num_questions=num_questions,
        num_choices=n_choices,
        seed=seed,
    )

    typer.echo(f"Question run written to {run_path}")
    for record, out in results:
        write_prompt(out)
        typer.echo(
            f"Question {record['question_id']} type={record['type']} "
            f"varying={record['varying_axes']} correct={record['correct_letter']} at {out}"
        )


if __name__ == "__main__":
    app()
