#!/usr/bin/env python3
"""P1: sample random MLP/KAN settings on a fixed dataset and write a candidate set + GT.

Uses ArchitectureIQ's generate_candidate_set (vary model only by default).
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--profile", default="v2.5-xor-holdout")
    parser.add_argument("--budget", type=int, default=8192)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--vary",
        default="model",
        help="Comma-separated axes: model,optimizer,loss",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from architecture_iq.candidates.sets import generate_candidate_set
    from architecture_iq.profile import load_profile
    from architecture_iq.registry import ensure_registries

    ensure_registries()
    profile = load_profile(args.profile)
    vary = frozenset(a.strip() for a in args.vary.split(",") if a.strip())
    rng = random.Random(args.seed)

    print(
        f"profile={profile.name} dataset={args.dataset} "
        f"count={args.count} budget={args.budget} vary={sorted(vary)} device={args.device}"
    )
    if args.dry_run:
        return

    set_dir = generate_candidate_set(
        profile,
        dataset_path=args.dataset.resolve(),
        budget=args.budget,
        count=args.count,
        varying_axes=vary,
        rng=rng,
        seed=args.seed,
        execution_device=args.device,
    )
    print(f"candidate set → {set_dir}")


if __name__ == "__main__":
    main()
