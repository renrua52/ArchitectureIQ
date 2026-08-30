"""xor_classification / spiral_classification as dataset families of their own.

Before v1.4 both were `rule_family` values inside
synthetic_tabular_classification, so all five rules shared one benchmark bucket
whose difficulty depended on which rule the seed happened to draw. These tests
pin the split: the two new families exist, they cannot be talked out of their
rule, they get their own dataset_id prefix and prompt template, and the generic
pipeline (train.py choice, bucketing) follows the family instead of a hardcoded
name list. `tests/test_xor_spiral_classification.py` covers the legacy path,
which still has to work for artifacts already on disk.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from architecture_iq.candidates.generator import _train_py_for_family
from architecture_iq.datasets import create_dataset
from architecture_iq.paths import PROMPTS_DIR
from architecture_iq.profile import load_profile
from architecture_iq.registry import ensure_registries, get_dataset_family, list_dataset_families
from architecture_iq.runtime.loader import load_synthesize_module

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from benchmark_v1_build import dataset_bucket  # noqa: E402

PROFILE = "v1.4"


@pytest.mark.parametrize(
    ("family_name", "prefix", "rule", "options"),
    [
        ("xor_classification", "xorcls_", "xor", ("input_dim",)),
        ("spiral_classification", "spiralcls_", "spiral", ()),
    ],
)
def test_new_families_declare_their_rule_and_prefix(
    family_name: str, prefix: str, rule: str, options: tuple[str, ...]
) -> None:
    ensure_registries()
    assert family_name in list_dataset_families()
    family = get_dataset_family(family_name)
    assert family.forced_rule_family == rule
    assert family.supported_rule_families == (rule,)
    assert family.dataset_id_prefix == prefix.rstrip("_")
    assert family.instance_option_names == options
    # Both reuse the classification training loop, whose train_and_eval returns
    # the final_test_ce key selection_metric_name() asks for.
    assert family.train_loop_kind == "classification"
    assert family.selection_metric_name() == "test_ce"


def _build(profile, family_name: str, seed: int, out_dir: Path) -> dict:
    """Spec + materialized files in a temp dir (create_dataset writes into data/)."""
    family = get_dataset_family(family_name)
    partial = family.create_instance(profile, seed)
    spec = family.build_spec_with_id(partial)
    family.materialize({**partial, **spec}, out_dir)
    return {**partial, **spec}


@pytest.mark.parametrize(
    ("family_name", "prefix"),
    [("xor_classification", "xorcls_"), ("spiral_classification", "spiralcls_")],
)
def test_new_families_materialize_under_their_own_id(
    family_name: str, prefix: str, tmp_path: Path
) -> None:
    ensure_registries()
    profile = load_profile(PROFILE)
    spec = _build(profile, family_name, 5, tmp_path / family_name)
    assert spec["dataset_id"].startswith(prefix)
    assert spec["family"] == family_name
    assert spec["params"]["rule_family"] == get_dataset_family(family_name).forced_rule_family
    module = load_synthesize_module(tmp_path / family_name / "synthesize.py")
    train_x, train_y, _, _ = module.synthesize()
    assert train_x.shape == (spec["params"]["train_size"], spec["params"]["input_dim"])
    assert set(train_y.unique().tolist()) <= {0, 1}


def test_new_family_ids_differ_from_the_shared_family_at_the_same_seed(
    tmp_path: Path,
) -> None:
    """Separate buckets must not collide on disk with the tabular family."""
    ensure_registries()
    profile = load_profile(PROFILE)
    ids = {
        family_name: _build(profile, family_name, 3, tmp_path / family_name)["dataset_id"]
        for family_name in (
            "xor_classification",
            "spiral_classification",
            "synthetic_tabular_classification",
        )
    }
    assert len(set(ids.values())) == 3, ids


def test_new_families_reject_a_rule_family_option() -> None:
    """The rule is the family, so --rule-family has nothing to choose."""
    profile = load_profile(PROFILE)
    for family_name in ("xor_classification", "spiral_classification"):
        with pytest.raises(ValueError, match="rule_family"):
            create_dataset(
                profile,
                0,
                family_name=family_name,
                family_options={"rule_family": "smooth_additive"},
            )


def test_spiral_rejects_an_input_dim_option() -> None:
    """Spiral is 2-D by construction; xor still takes a dimension."""
    profile = load_profile(PROFILE)
    with pytest.raises(ValueError, match="input_dim"):
        create_dataset(
            profile,
            0,
            family_name="spiral_classification",
            family_options={"input_dim": 4},
        )
    assert "input_dim" in get_dataset_family("xor_classification").instance_option_names


def test_profile_config_cannot_contradict_the_forced_rule() -> None:
    profile = load_profile(PROFILE)
    profile.raw["dataset_configs"]["xor_classification"]["rule_families"] = ["spiral"]
    family = get_dataset_family("xor_classification")
    with pytest.raises(ValueError, match="always uses rule_family"):
        family.create_instance(profile, 0)


def test_generated_train_py_is_the_classification_loop() -> None:
    ensure_registries()
    for family_name in ("xor_classification", "spiral_classification"):
        train_py = _train_py_for_family(family_name)
        assert "final_test_ce" in train_py
        assert "final_test_accuracy" in train_py
        assert "final_test_mse" not in train_py


def test_dataset_bucket_splits_the_new_families_and_still_reads_legacy_specs() -> None:
    assert dataset_bucket("xor_classification", {}) == "xor"
    assert dataset_bucket("spiral_classification", {}) == "spiral"
    assert dataset_bucket("synthetic_tabular_classification", {}) == "general_tabular"
    # Artifacts generated before the split carry the rule in params.
    for rule, bucket in (("xor", "xor"), ("spiral", "spiral"), ("smooth_additive", "general_tabular")):
        legacy = {"params": {"rule_family": rule}}
        assert dataset_bucket("synthetic_tabular_classification", legacy) == bucket


def test_every_registered_family_has_a_dataset_prompt_template() -> None:
    """A missing template silently falls back to the univariate blurb."""
    ensure_registries()
    for family_name in list_dataset_families():
        template = PROMPTS_DIR / "dataset" / f"{family_name}.md"
        assert template.exists(), template
        assert template.read_text(encoding="utf-8").strip()
