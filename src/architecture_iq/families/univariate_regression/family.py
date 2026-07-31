from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
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
    label_noise_std: float = {label_noise_std},
    label_noise_seed: int = {label_noise_seed},
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(point_seed)
    train_x = torch.rand(train_size, generator=gen) * (domain_high - domain_low) + domain_low
    test_x = torch.rand(test_size, generator=gen) * (domain_high - domain_low) + domain_low
    train_y = target(train_x)
    test_y = target(test_x)
    # Label noise is added to TRAINING targets only; the test split stays the
    # exact target function, so test error measures recovery of the true signal.
    if label_noise_std > 0.0:
        noise_gen = torch.Generator().manual_seed(label_noise_seed)
        train_y = train_y + label_noise_std * torch.randn(
            train_y.shape, generator=noise_gen
        )
    return train_x.unsqueeze(-1), train_y.unsqueeze(-1), test_x.unsqueeze(-1), test_y.unsqueeze(-1)


if __name__ == "__main__":
    tx, ty, vx, vy = synthesize()
    print("train", tx.shape, ty.shape, "test", vx.shape, vy.shape)
'''


class UnivariateRegressionFamily(DatasetFamily):
    name = "univariate_regression"

    @staticmethod
    def _rng_streams(instance_seed: int) -> tuple[int, int, int]:
        """Derive internal expression, point-sampling, and label-noise seeds."""
        expression_seed = instance_seed
        point_seed = instance_seed + 1_000
        label_noise_seed = instance_seed + 2_000
        return expression_seed, point_seed, label_noise_seed

    def create_instance(
        self,
        profile: Profile,
        seed: int,
        *,
        noise_std: float | None = None,
    ) -> dict[str, Any]:
        expression_seed, point_seed, label_noise_seed = self._rng_streams(seed)
        cfg = profile.family_config(self.name)
        domain = tuple(cfg["domain"])
        sampler_cfg = cfg["sampler"]
        sampled = sample_symbolic_expression(
            seed=expression_seed,
            max_depth=int(sampler_cfg["max_depth"]),
            max_retries=int(sampler_cfg.get("max_retries", 200)),
            domain=(float(domain[0]), float(domain[1])),
        )
        noise_std = float(noise_std) if noise_std is not None else 0.0
        if noise_std < 0.0:
            raise ValueError("noise_std must be >= 0")
        noise = (
            {"enabled": True, "type": "gaussian_label", "std": noise_std,
             "seed": label_noise_seed, "applies_to": "train_only"}
            if noise_std > 0.0
            else {"enabled": False}
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
            "noise": noise,
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
        spec["quality_tags"] = self._compute_quality_tags(spec)
        return spec

    @staticmethod
    def _compute_quality_tags(spec: dict[str, Any]) -> list[str]:
        """Compute quality tags for the dataset instance.

        Currently checks for affine degeneracy (target function is near-linear).
        """
        tags: list[str] = []
        try:
            torch_expr = spec.get("_torch_expression")
            if not torch_expr:
                return tags
            domain = spec["params"]["domain"]
            x = np.linspace(domain[0], domain[1], 256)
            # Evaluate the target on a dense grid via numpy
            import re
            expr = spec["params"]["expression"]
            # Quick numeric evaluation via a simple approach
            # Use the torch expression string to build a lambda
            ns = {"np": np, "sin": np.sin, "cos": np.cos, "tanh": np.tanh, "abs": np.abs, "pi": np.pi}
            # Replace torch.* with numpy.*
            py_expr = torch_expr
            py_expr = py_expr.replace("torch.sin", "np.sin")
            py_expr = py_expr.replace("torch.cos", "np.cos")
            py_expr = py_expr.replace("torch.tanh", "np.tanh")
            py_expr = py_expr.replace("torch.abs", "np.abs")
            py_expr = py_expr.replace("torch.tensor(", "")
            py_expr = re.sub(r",\s*dtype=x\.dtype,\s*device=x\.device\)", "", py_expr)
            py_expr = py_expr.replace("torch.clamp(", "np.clip(")
            py_expr = py_expr.replace("torch.sign(", "np.sign(")
            # Keep the expression as-is, it should be evaluable with numpy
            import ast
            try:
                tree = ast.parse(py_expr, mode="eval")
                code = compile(tree, "<expr>", "eval")
                y = eval(code, {"np": np, "x": x, "pi": np.pi})
                y = np.asarray(y, dtype=float).ravel()
            except Exception:
                return tags

            if not np.all(np.isfinite(y)):
                return tags

            # Affine fit
            x_design = np.column_stack([np.ones(len(x)), x])
            coef, *_ = np.linalg.lstsq(x_design, y, rcond=None)
            pred = x_design @ coef
            ss_res = float(np.sum((y - pred) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

            if np.isfinite(r2) and r2 > 0.95:
                tags.append("degraded_linear")

            # Check y_range
            y_range = float(np.max(y) - np.min(y))
            if y_range < 0.4:
                tags.append("degraded_low_range")

        except Exception:
            pass

        return tags

    def materialize(self, spec: dict[str, Any], out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        params = spec["params"]
        domain = params["domain"]
        torch_expr = spec.get("_torch_expression")
        if not torch_expr:
            raise ValueError("Missing _torch_expression for materialization")
        noise = params.get("noise", {"enabled": False})
        noise_std = float(noise.get("std", 0.0)) if noise.get("enabled") else 0.0
        noise_seed = int(noise.get("seed", params["instance_seed"] + 2_000))
        synth_code = SYNTHESIZE_TEMPLATE.format(
            torch_expr=torch_expr,
            train_size=params["train_size"],
            test_size=params["test_size"],
            point_seed=params["point_sampling"]["seed"],
            domain_low=domain[0],
            domain_high=domain[1],
            label_noise_std=noise_std,
            label_noise_seed=noise_seed,
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
