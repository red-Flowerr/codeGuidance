from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass(slots=True)
class DatasetRecord:
    """Single dataset entry that will be passed to the LLM."""

    identifier: str
    prompt: str
    reference: Optional[str] = None
    language: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_payload(payload: Dict[str, Any]) -> "DatasetRecord":
        identifier = str(
            payload.get("id")
            or payload.get("identifier")
            or payload.get("name")
            or payload.get("task_id")
        )
        if not identifier:
            raise ValueError("Dataset record must contain a non-empty id field.")

        prompt = payload.get("prompt")
        if not prompt:
            raise ValueError(f"Dataset record {identifier} is missing a 'prompt'.")

        reference = payload.get("reference") or payload.get("ground_truth")
        language = payload.get("language")
        metadata = {
            key: value
            for key, value in payload.items()
            if key not in {"id", "identifier", "name", "task_id", "prompt", "reference", "ground_truth", "language"}
        }

        return DatasetRecord(
            identifier=identifier,
            prompt=prompt,
            reference=reference,
            language=language,
            metadata=metadata,
        )


def load_jsonl_dataset(path: str | Path) -> List[DatasetRecord]:
    """Load dataset records from a JSONL file."""
    records: List[DatasetRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            records.append(DatasetRecord.from_payload(payload))
    return records


def iter_dataset(records: Iterable[DatasetRecord]) -> Iterable[DatasetRecord]:
    """Simple passthrough iterator, useful for swapping to streaming sources later."""
    for record in records:
        yield record
