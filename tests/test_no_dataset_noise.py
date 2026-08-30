"""No v1.4 dataset carries noise -- in the spec, the code, or the prompt.

Three layers have to agree, and each used to name noise separately: the frozen
``dataset_spec.json``, the generated ``synthesize.py`` that actually produces the
data, and the prompt the model reads. A dataset with no noise must not mention it
anywhere -- not even to say there is none.

The gate is the profile: a family config without ``noise_std`` produces no noise
at all. Profiles that still declare it (v1.3, which produced the existing
benchmark build) must keep generating exactly what they did before, so the second
half of this module pins that path down too.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

from architecture_iq.families.synthetic_tabular_classification import RULE_FAMILIES
from architecture_iq.profile import load_profile
from architecture_iq.prompts.code_excerpt import excerpt_synthesize_py
from architecture_iq.prompts.formatters import (
    format_dataset_protocol,
    format_regression_protocol,
    format_synthetic_tabular_classification_rule,
)
from architecture_iq.registry import ensure_registries, get_dataset_family

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "question_inspector"
sys.path.insert(0, str(TOOLS))
import prompt_format as insp  # noqa: E402

TEMPLATES = Path(__file__).resolve().parents[1] / "prompts" / "templates" / "dataset"

TABULAR_FAMILIES = frozenset(
    {"synthetic_tabular_classification", "xor_classification", "spiral_classification"}
)

# "jitter" and a bare epsilon are the same claim under another name, so they are
# part of what must be gone.
NOISE_WORDS = re.compile(r"nois|jitter|varepsilon|ε", re.IGNORECASE)


@pytest.fixture(scope="module")
def v14():
    ensure_registries()
    return load_profile("v1.4")


def _families(profile) -> list[str]:
    return list(profile.pools["dataset_families"])


def test_v14_declares_no_noise_for_any_family(v14) -> None:
    for family in _families(v14):
        assert "noise_std" not in v14.family_config(family), family


@pytest.mark.parametrize("seed", [0, 1])
def test_v14_spec_code_and_prompt_never_mention_noise(v14, tmp_path, seed) -> None:
    for family in _families(v14):
        spec = get_dataset_family(family).create_instance(v14, seed)
        out = tmp_path / family / str(seed)
        get_dataset_family(family).materialize(spec, out)
        params = spec["params"]

        layers = {
            "spec": json.dumps(spec),
            "synthesize.py": (out / "synthesize.py").read_text(encoding="utf-8"),
            "dataset template": (TEMPLATES / f"{family}.md").read_text(encoding="utf-8"),
            "protocol section": format_dataset_protocol(params, family=family),
        }
        if family in TABULAR_FAMILIES:
            layers["rule card"] = format_synthetic_tabular_classification_rule(params)
        else:
            # Regression / LM prompts show the synthesis code itself, so the
            # excerpt is the layer the model actually reads.
            layers["synthesis excerpt"] = excerpt_synthesize_py(layers["synthesize.py"])

        for layer, text in layers.items():
            match = NOISE_WORDS.search(text)
            assert match is None, f"{family} seed {seed} {layer}: {match.group(0)!r}"


def test_v14_classification_labels_are_an_exact_function_of_x(v14, tmp_path) -> None:
    """Not just unmentioned: the labels really are deterministic now.

    Re-running the generated ``synthesize`` gives the same labels, and for the
    score rules the label matches ``s(x) > threshold`` on every row -- which is
    exactly what fails when a noise term is still added.
    """
    from architecture_iq.runtime.loader import load_synthesize_module

    for family in sorted(TABULAR_FAMILIES):
        spec = get_dataset_family(family).create_instance(v14, 0)
        out = tmp_path / "exact" / family
        get_dataset_family(family).materialize(spec, out)
        module = load_synthesize_module(out / "synthesize.py")
        train_x, train_y, _, _ = module.synthesize()
        again_x, again_y, _, _ = module.synthesize()
        assert (again_x == train_x).all(), family
        assert (again_y == train_y).all(), family
        if spec["params"]["rule_family"] == "spiral":
            continue
        threshold = float(spec["params"]["decision_threshold"])
        expected = (module.target(train_x) > threshold).to(train_y.dtype)
        assert (expected == train_y).all(), family


def test_v14_regression_specs_drop_the_noise_field(v14, tmp_path) -> None:
    for family in ("univariate_regression", "multivariate_regression"):
        spec = get_dataset_family(family).create_instance(v14, 0)
        assert "noise" not in spec["params"], family


# --- the noisy path, which older profiles still take ------------------------


@pytest.mark.parametrize("seed", range(6))
def test_v13_tabular_still_generates_noise(seed, tmp_path) -> None:
    ensure_registries()
    profile = load_profile("v1.3")
    family = get_dataset_family("synthetic_tabular_classification")
    spec = family.create_instance(profile, seed)
    out = tmp_path / str(seed)
    family.materialize(spec, out)
    code = (out / "synthesize.py").read_text(encoding="utf-8")

    noise_std = spec["params"]["noise_std"]
    assert noise_std > 0.0
    assert f"    noise_std: float = {noise_std!r}," in code
    if spec["params"]["rule_family"] == "spiral":
        assert "    noise_std: float,\n" in code
        assert " noise_std=noise_std, seed=point_seed\n" in code
        assert " noise_std=noise_std, seed=point_seed + 1\n" in code
        assert (
            "return points + noise_std * torch.randn(count, 2, generator=gen)" in code
        )
    else:
        assert (
            "train_score = target(train_x) + noise_std"
            " * torch.randn(train_size, generator=gen)" in code
        )
        assert (
            "test_score = target(test_x) + noise_std"
            " * torch.randn(test_size, generator=gen)" in code
        )


# One weight per term, and the term count is rule-specific: two active features
# for the additive rule, one pair for the interaction rule, three branch weights
# for the piecewise rule.
RULE_WEIGHTS = {
    "smooth_additive": [-1.0, 0.75],
    "sparse_interaction": [-1.0],
    "piecewise_boundary": [-1.0, 0.75, 0.5],
    "xor": [-1.0],
}


@pytest.mark.parametrize("rule_family", [*RULE_FAMILIES, "xor"])
def test_rule_card_describes_noise_only_when_the_spec_has_it(rule_family) -> None:
    params = {
        "input_dim": 4,
        "rule_family": rule_family,
        "active_features": [0, 2],
        "interaction_pairs": [[0, 2]],
        "rule_weights": RULE_WEIGHTS[rule_family],
        "piecewise_breakpoint": -0.25,
        "decision_threshold": 0.125,
        "point_sampling": {"seed": 11},
        "calibration": {"seed": 22, "size": 4096, "target_positive_rate": 0.5},
    }
    quiet = format_synthetic_tabular_classification_rule(params)
    loud = format_synthetic_tabular_classification_rule({**params, "noise_std": 0.1})

    assert NOISE_WORDS.search(quiet) is None
    assert "Label noise" in loud
    assert "s(x) + ε" in loud
    assert "point/noise seed" in loud
    assert "point seed" in quiet
    # The rule itself still has to be stated either way.
    for text in (quiet, loud):
        assert "Latent score" in text
        assert "Label rule" in text
        assert "Bayes decision boundary" in text
    # And the inspector mirror must agree on both branches.
    assert quiet == insp.format_synthetic_tabular_classification_rule(params)
    assert loud == insp.format_synthetic_tabular_classification_rule(
        {**params, "noise_std": 0.1}
    )


def test_spiral_card_describes_jitter_only_when_the_spec_has_it() -> None:
    params = {
        "input_dim": 2,
        "rule_family": "spiral",
        "active_features": [0, 1],
        "interaction_pairs": [],
        "rule_weights": [1.0],
        "piecewise_breakpoint": 0.0,
        "spiral_turns": 2.0,
        "decision_threshold": 0.0,
        "point_sampling": {"distribution": "two_spirals", "seed": 11, "turns": 2.0},
        "calibration": {"seed": 22, "size": 0, "target_positive_rate": 0.5},
    }
    quiet = format_synthetic_tabular_classification_rule(params)
    loud = format_synthetic_tabular_classification_rule({**params, "noise_std": 0.05})

    assert NOISE_WORDS.search(quiet) is None
    assert "is then added to each coordinate" in loud
    assert "two noiseless arms" in loud
    assert "the two arms" in quiet
    for text in (quiet, loud):
        assert "two-spirals" in text
        assert "phase = π" in text
    assert quiet == insp.format_synthetic_tabular_classification_rule(params)


def test_regression_protocol_states_noise_only_for_a_legacy_noisy_spec() -> None:
    params = {
        "train_size": 256,
        "test_size": 256,
        "domain": [0.0, 1.0],
        "expression": "x**2",
        "point_sampling": {"seed": 3},
    }
    quiet = format_regression_protocol(params)
    assert NOISE_WORDS.search(quiet) is None
    assert "Target expression" in quiet

    legacy = format_regression_protocol(
        {**params, "noise": {"enabled": True, "sigma": 0.05}}
    )
    assert "Gaussian observation noise with sigma=0.05" in legacy

    # A pre-v1.4 spec that recorded noise as explicitly disabled says nothing
    # about it either -- the absence is not worth a line in the prompt.
    disabled = format_regression_protocol({**params, "noise": {"enabled": False}})
    assert NOISE_WORDS.search(disabled) is None
    assert disabled == quiet
