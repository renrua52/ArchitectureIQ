from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch.nn as nn


class ModelFamily(ABC):
    name: str

    @abstractmethod
    def validate(self, model_spec: dict[str, Any]) -> None:
        ...

    @abstractmethod
    def build_module(self, model_spec: dict[str, Any]) -> nn.Module:
        ...

    @abstractmethod
    def render_model_py(self, model_spec: dict[str, Any]) -> str:
        ...

    @abstractmethod
    def sample_spec(
        self,
        profile: Any,
        rng: Any,
        dataset_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ...
