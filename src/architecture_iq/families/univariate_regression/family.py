from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from architecture_iq.families.base import DatasetFamily
from architecture_iq.families.univariate_regression.sampler import sample_symbolic_expression
from architecture_iq.profile import Profile
from architecture_iq.util import short_hash, write_json


SYNTHESIZE_TEMPLATE = '''"""Dataset synthesis — source of truth for this instance."""
from __future__ import annotations

import torch


def target(x: torch.Tensor) -> torch.Tensor:
    return {torch_expr}


def synthesize(
    *,
    train_size: int = {train_size},
    test_size: int = {test_size},
    point_seed: int = {point_seed},
    domain_low: float = {domain_low},
    domain_high: float = {domain_high},
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(point_seed)
    train_x = torch.rand(train_size, generator=gen) * (domain_high - domain_low) + domain_low
    test_x = torch.rand(test_size, generator=gen) * (domain_high - domain_low) + domain_low
    train_y = target(train_x)
    test_y = target(test_x)
    return train_x.unsqueeze(-1), train_y.unsqueeze(-1), test_x.unsqueeze(-1), test_y.unsqueeze(-1)


if __name__ == "__main__":
    tx, ty, vx, vy = synthesize()
    print("train", tx.shape, ty.shape, "test", vx.shape, vy.shape)
'''


class UnivariateRegressionFamily(DatasetFamily):
    name = "univariate_regression"

    @staticmethod
    def _rng_streams(instance_seed: int) -> tuple[int, int]:
        """Derive internal expression and point-sampling seeds from one instance seed."""
        expression_seed = instance_seed
        point_seed = instance_seed + 1_000
        return expression_seed, point_seed

    def create_instance(
        self,
        profile: Profile,
        seed: int,
    ) -> dict[str, Any]:
        expression_seed, point_seed = self._rng_streams(seed)
        cfg = profile.dataset
        domain = tuple(cfg["domain"])
        sampler_cfg = cfg["sampler"]
        sampled = sample_symbolic_expression(
            seed=expression_seed,
            max_depth=int(sampler_cfg["max_depth"]),
            max_retries=int(sampler_cfg.get("max_retries", 200)),
            domain=(float(domain[0]), float(domain[1])),
        )
        params = {
            "instance_seed": seed,
            "sampler": {
                "id": sampler_cfg["id"],
                "seed": expression_seed,
                "max_depth": int(sampler_cfg["max_depth"]),
                "retry": sampled.retry,
            },
            "expression": sampled.expression,
            "domain": list(domain),
            "train_size": int(cfg["train_size"]),
            "test_size": int(cfg["test_size"]),
            "noise": {"enabled": False},
            "point_sampling": {"distribution": "uniform", "seed": point_seed},
        }
        significance = {
            "gap_min": float(profile.significance["gap_min"]),
            "fail_threshold": float(profile.ground_truth["fail_threshold"]),
        }
        return {
            "schema_version": profile.schema_version,
            "family": self.name,
            "params": params,
            "selection_metric": "test_mse",
            "significance": significance,
            "files": {
                "synthesize": "synthesize.py",
                "train": "train.pt",
                "test": "test.pt",
            },
            "_torch_expression": sampled.torch_expression,
        }

    def build_spec_with_id(self, partial: dict[str, Any]) -> dict[str, Any]:
        dataset_id = f"sym_{short_hash(partial['params'])}"
        spec = {k: v for k, v in partial.items() if not k.startswith("_")}
        spec["dataset_id"] = dataset_id
        return spec

    def materialize(self, spec: dict[str, Any], out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        params = spec["params"]
        domain = params["domain"]
        torch_expr = spec.get("_torch_expression")
        if not torch_expr:
            raise ValueError("Missing _torch_expression for materialization")
        synth_code = SYNTHESIZE_TEMPLATE.format(
            torch_expr=torch_expr,
            train_size=params["train_size"],
            test_size=params["test_size"],
            point_seed=params["point_sampling"]["seed"],
            domain_low=domain[0],
            domain_high=domain[1],
        )
        (out_dir / "synthesize.py").write_text(synth_code, encoding="utf-8")

        from architecture_iq.runtime.loader import load_synthesize_module

        module = load_synthesize_module(out_dir / "synthesize.py")
        tx, ty, vx, vy = module.synthesize()

        torch.save({"x": tx, "y": ty}, out_dir / "train.pt")
        torch.save({"x": vx, "y": vy}, out_dir / "test.pt")

        write_json(out_dir / "dataset_spec.json", {k: v for k, v in spec.items() if not k.startswith("_")})

    def load_tensors(self, dataset_path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        train = torch.load(dataset_path / "train.pt", weights_only=True)
        test = torch.load(dataset_path / "test.pt", weights_only=True)
        return train["x"], train["y"], test["x"], test["y"]

    def selection_metric_name(self) -> str:
        return "test_mse"

    def default_significance(self) -> dict[str, Any]:
        return {}

    def compatible_model_types(self) -> list[str]:
        return ["mlp"]
