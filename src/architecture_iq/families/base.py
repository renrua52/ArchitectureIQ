from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from architecture_iq.profile import Profile


class DatasetFamily(ABC):
    name: str

    @abstractmethod
    def create_instance(
        self,
        profile: Profile,
        seed: int,
    ) -> dict[str, Any]:
        """Return dataset_spec dict (without dataset_id).

        ``seed`` is the single user-facing instance seed; each family derives
        any internal RNG streams from it.
        """

    @abstractmethod
    def materialize(self, spec: dict[str, Any], out_dir: Path) -> None:
        """Write synthesize.py, tensors, and dataset_spec.json to out_dir."""

    @abstractmethod
    def load_tensors(self, dataset_path: Path) -> tuple[Any, Any, Any, Any]:
        """Return train_x, train_y, test_x, test_y tensors."""

    @abstractmethod
    def selection_metric_name(self) -> str:
        ...

    @abstractmethod
    def default_significance(self) -> dict[str, Any]:
        ...

    @abstractmethod
    def compatible_model_types(self) -> list[str]:
        ...
