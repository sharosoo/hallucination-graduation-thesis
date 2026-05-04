"""Corpus statistics port."""

from __future__ import annotations

from abc import ABC, abstractmethod

from experiments.domain import CorpusStats, ModelResponse, QuestionExample


class CorpusStatsPort(ABC):
    """Fetches corpus-grounded statistics for a question/response pair."""

    @abstractmethod
    def get_corpus_stats(
        self,
        example: QuestionExample,
        response: ModelResponse,
    ) -> CorpusStats:
        """Return typed corpus statistics for the sample."""
