from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional

from .dataset import DatasetRecord
from .llm_client import LLMGeneration, LLMToken


PLACEHOLDER_PATTERN = re.compile(r"\b(?:TODO|FIXME|TBD|\.\.\.)\b", re.IGNORECASE)


def compute_latency_metric(generation: LLMGeneration) -> Dict[str, float]:
    return {"latency_ms": generation.latency_ms}


def compute_length_metrics(generation: LLMGeneration) -> Dict[str, Optional[int]]:
    text = generation.text or ""
    token_count = len(generation.tokens) if generation.tokens else None
    line_count = text.count("\n") + (1 if text else 0)
    return {
        "response_char_length": len(text),
        "response_token_count": token_count,
        "response_line_count": line_count,
    }


def compute_placeholder_metrics(generation: LLMGeneration) -> Dict[str, Any]:
    text = generation.text or ""
    matches = PLACEHOLDER_PATTERN.findall(text)
    return {
        "placeholder_count": len(matches),
        "has_placeholders": bool(matches),
    }


def compute_entropy_metrics(generation: LLMGeneration) -> Dict[str, Any]:
    entropies: List[float] = []
    for token in generation.tokens:
        entropy = _token_entropy(token)
        if entropy is not None:
            entropies.append(entropy)

    if not entropies:
        return {
            "mean_token_entropy": None,
            "max_token_entropy": None,
            "token_entropies": [],
        }

    mean_entropy = sum(entropies) / len(entropies)
    return {
        "mean_token_entropy": mean_entropy,
        "max_token_entropy": max(entropies),
        "token_entropies": entropies,
    }


def compute_syntax_validity(generation: LLMGeneration, record: DatasetRecord) -> Dict[str, Any]:
    language = (record.language or "").lower()
    code = generation.text or ""

    if not code.strip():
        return {"syntax_valid": False, "syntax_error": "Empty response."}

    if language in {"python", "py", ""}:
        try:
            compile(code, "<string>", "exec")
        except SyntaxError as exc:
            return {"syntax_valid": False, "syntax_error": str(exc)}
        return {"syntax_valid": True, "syntax_error": None}

    # Fallback: we cannot validate other languages reliably without tooling.
    return {"syntax_valid": None, "syntax_error": None}


def compute_exact_match(generation: LLMGeneration, record: DatasetRecord) -> Dict[str, Any]:
    if record.reference is None:
        return {"exact_match": None}
    normalized_response = generation.text.strip()
    normalized_reference = record.reference.strip()
    return {"exact_match": normalized_response == normalized_reference}


def _token_entropy(token: LLMToken) -> Optional[float]:
    """Compute entropy for a token using available logprobs."""
    candidates: Dict[str, float] = {}
    if token.top_logprobs:
        candidates.update(token.top_logprobs)
    # Ensure the chosen token is included.
    candidates[token.text] = token.logprob

    probs = [math.exp(logprob) for logprob in candidates.values()]
    total = sum(probs)
    if total <= 0:
        return None

    normalized = [prob / total for prob in probs if prob > 0]
    if not normalized:
        return None

    return -sum(prob * math.log(prob) for prob in normalized)
