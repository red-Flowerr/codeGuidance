"""Utilities for running LLM prompts over code datasets and collecting metrics."""

from .dataset import DatasetRecord, load_jsonl_dataset
from .llm_client import LLMClient, OpenAIClient, LLMToken, LLMGeneration
from .metrics import (
    compute_latency_metric,
    compute_length_metrics,
    compute_entropy_metrics,
    compute_syntax_validity,
)
from .runner import LLMObserverRunner, ObserverConfig

__all__ = [
    "DatasetRecord",
    "load_jsonl_dataset",
    "LLMClient",
    "OpenAIClient",
    "LLMToken",
    "LLMGeneration",
    "compute_latency_metric",
    "compute_length_metrics",
    "compute_entropy_metrics",
    "compute_syntax_validity",
    "LLMObserverRunner",
    "ObserverConfig",
]
