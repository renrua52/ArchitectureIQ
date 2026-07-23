from __future__ import annotations

from pathlib import Path

from architecture_iq.prompts.code_excerpt import (
    excerpt_loss_py,
    excerpt_model_py,
    excerpt_optimizer_py,
    excerpt_synthesize_py,
)
from architecture_iq.prompts.renderer import render_prompt


SAMPLE_MODEL = '''"""docstring"""
from __future__ import annotations
import torch
import torch.nn as nn

def _activation(name: str) -> nn.Module:
    return nn.ReLU()

class MLPBlock(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

class Model(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
'''

SAMPLE_LOSS = '''import torch
import torch.nn as nn

def loss_fn(model: nn.Module, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)
'''

SAMPLE_OPT = '''import torch.nn as nn

def build_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.SGD(model.parameters(), lr=0.01)
'''

SAMPLE_SYNTH = '''import torch

def target(x: torch.Tensor) -> torch.Tensor:
    return x ** 2

def synthesize():
    pass

if __name__ == "__main__":
    synthesize()
'''


def test_excerpt_model_py() -> None:
    out = excerpt_model_py(SAMPLE_MODEL)
    assert "import torch" not in out
    assert "def _activation" in out
    assert "class MLPBlock(nn.Module):" in out
    assert "class Model(nn.Module):" in out


def test_excerpt_loss_py() -> None:
    out = excerpt_loss_py(SAMPLE_LOSS)
    assert "import torch" not in out
    assert "def loss_fn" in out


def test_excerpt_optimizer_py() -> None:
    out = excerpt_optimizer_py(SAMPLE_OPT)
    assert "import torch" not in out
    assert "def build_optimizer" in out


def test_excerpt_synthesize_py() -> None:
    out = excerpt_synthesize_py(SAMPLE_SYNTH)
    assert "import torch" not in out
    assert "if __name__" not in out
    assert "def target" in out
    assert "def synthesize" in out


def test_excerpt_synthesize_py_bigram_lm() -> None:
    from architecture_iq.families.bigram_lm.family import SYNTHESIZE_TEMPLATE

    source = SYNTHESIZE_TEMPLATE.format(
        train_size=100,
        test_size=50,
        sequence_seed=1,
        table_seed=2,
        vocab_size=16,
        context_length=8,
        alpha=1.0,
        layout="lm",
    )
    out = excerpt_synthesize_py(source)
    assert "def target" in out
    assert "def build_transition(" in out
    assert "def synthesize(" in out
    assert "seq[:, 1:]" in out
    assert "if __name__" not in out


def test_render_prompt_uses_excerpts() -> None:
    q_path = Path("data/questions/q_605ae4")
    if not q_path.exists():
        return
    text = render_prompt(q_path)
    assert "if __name__" not in text
    assert "from __future__ import annotations" not in text
    assert "def loss_fn" in text
    assert "class Model(nn.Module):" in text
    assert "def target" in text
