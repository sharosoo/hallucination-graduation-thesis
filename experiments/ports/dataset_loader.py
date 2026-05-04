"""Dataset loading port."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from experiments.domain import QuestionExample


class DatasetLoaderPort(ABC):
    """Loads canonical question examples without binding to a concrete source."""

    @abstractmethod
    def load_examples(self, dataset_name: str, split_id: str) -> Sequence[QuestionExample]:
        """Load typed question examples for a dataset split."""
