from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from .dataset import DatasetRecord, iter_dataset
from .llm_client import LLMClient, LLMGeneration
from .metrics import (
    compute_entropy_metrics,
    compute_exact_match,
    compute_latency_metric,
    compute_length_metrics,
    compute_placeholder_metrics,
    compute_syntax_validity,
)


MetricFunction = Callable[[LLMGeneration, DatasetRecord], Dict[str, Any]]


@dataclass(slots=True)
class ObserverConfig:
    """Configuration controlling how observations are stored."""

    save_path: Optional[Path] = None
    include_prompt: bool = False
    include_tokens: bool = False
    include_raw_response: bool = False
    extra_request_args: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Observation:
    record_id: str
    metrics: Dict[str, Any]
    response_text: str
    metadata: Dict[str, Any]

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "id": self.record_id,
            "response": self.response_text,
            "metrics": self.metrics,
            "metadata": self.metadata,
        }


class LLMObserverRunner:
    """Runs a dataset through an LLM client and captures observation metrics."""

    def __init__(
        self,
        client: LLMClient,
        *,
        config: Optional[ObserverConfig] = None,
        additional_metrics: Optional[List[MetricFunction]] = None,
        task_evaluator: Optional[MetricFunction] = None,
    ) -> None:
        self.client = client
        self.config = config or ObserverConfig()
        self.metric_functions: List[MetricFunction] = [
            lambda generation, record: compute_latency_metric(generation),
            lambda generation, record: compute_length_metrics(generation),
            lambda generation, record: compute_placeholder_metrics(generation),
            lambda generation, record: compute_entropy_metrics(generation),
            lambda generation, record: compute_syntax_validity(generation, record),
            lambda generation, record: compute_exact_match(generation, record),
        ]
        if additional_metrics:
            self.metric_functions.extend(additional_metrics)
        if task_evaluator:
            self.metric_functions.append(task_evaluator)

    def run(self, dataset: Iterable[DatasetRecord]) -> List[Observation]:
        observations: List[Observation] = []
        writer = self._create_writer() if self.config.save_path else None

        try:
            for record in iter_dataset(dataset):
                generation = self.client.generate(
                    record.prompt, **self.config.extra_request_args
                )
                metrics = self._collect_metrics(generation, record)
                metadata = self._build_metadata(generation, record)

                observation = Observation(
                    record_id=record.identifier,
                    metrics=metrics,
                    response_text=generation.text,
                    metadata=metadata,
                )

                observations.append(observation)
                if writer:
                    writer.write(json.dumps(observation.to_json_dict(), ensure_ascii=False))
                    writer.write("\n")
        finally:
            if writer:
                writer.close()

        return observations

    def _collect_metrics(
        self, generation: LLMGeneration, record: DatasetRecord
    ) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        for fn in self.metric_functions:
            metrics.update(fn(generation, record))
        return metrics

    def _build_metadata(
        self, generation: LLMGeneration, record: DatasetRecord
    ) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {"record_metadata": record.metadata}
        if self.config.include_prompt:
            metadata["prompt"] = record.prompt
        if self.config.include_tokens:
            metadata["tokens"] = [
                {
                    "text": token.text,
                    "logprob": token.logprob,
                    "top_logprobs": token.top_logprobs,
                }
                for token in generation.tokens
            ]
        if self.config.include_raw_response:
            metadata["raw_response"] = generation.raw_response
        return metadata

    def _create_writer(self):
        assert self.config.save_path is not None
        path = self.config.save_path
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.open("w", encoding="utf-8")
