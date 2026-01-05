from __future__ import annotations

import json
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from .dataset import DatasetRecord, iter_dataset
from .llm_client import LLMClient, LLMGeneration, LLMToken
from .metrics import compute_entropy_metrics


MetricFunction = Callable[[LLMGeneration, DatasetRecord], Dict[str, Any]]


@dataclass(slots=True)
class ObserverConfig:
    """Configuration controlling how observations are stored."""

    save_path: Optional[Path] = None
    include_prompt: bool = False
    include_tokens: bool = False
    include_raw_response: bool = False
    extra_request_args: Dict[str, Any] = field(default_factory=dict)
    html_report_path: Optional[Path] = None


@dataclass(slots=True)
class Observation:
    record_id: str
    metrics: Dict[str, Any]
    response_text: str
    tokens: List[LLMToken]
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
            lambda generation, record: compute_entropy_metrics(generation),
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
                    tokens=generation.tokens,
                    metadata=metadata,
                )

                observations.append(observation)
                if writer:
                    writer.write(json.dumps(observation.to_json_dict(), ensure_ascii=False))
                    writer.write("\n")
        finally:
            if writer:
                writer.close()

        if self.config.html_report_path:
            self._write_html_report(observations)

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
                    "is_special": token.is_special,
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

    def _write_html_report(self, observations: List[Observation]) -> None:
        assert self.config.html_report_path is not None
        path = self.config.html_report_path
        path.parent.mkdir(parents=True, exist_ok=True)

        def format_token_text(text: str) -> str:
            # Encode special characters so whitespace and control chars remain visible.
            display = text.encode("unicode_escape").decode("ascii")
            return escape(display)

        def sanitize_id(value: str, fallback: str) -> str:
            safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value)
            safe = safe.strip("-")
            return safe or fallback

        with path.open("w", encoding="utf-8") as handle:
            handle.write("<!DOCTYPE html><html><head><meta charset='utf-8'>")
            handle.write("<title>LLM Token Entropy Report</title>")
            handle.write(
                "<style>"
                "body{font-family:system-ui,-apple-system,\"Segoe UI\",sans-serif;background:#f5f6fb;color:#1f2530;margin:0;padding:24px;}"
                "h1{margin-top:0;font-size:24px;color:#182140;margin-bottom:24px;}"
                ".response-section{background:#fff;border:1px solid #e1e6f3;box-shadow:0 8px 16px rgba(15,23,42,0.06);border-radius:10px;padding:24px;margin-bottom:24px;}"
                ".record-title{margin:0 0 16px;font-size:18px;color:#273352;font-weight:600;}"
                ".prompt-block{margin-bottom:16px;padding:14px 16px;border:1px solid #dfe4f6;border-radius:8px;background:#f8faff;}"
                ".prompt-label{font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;color:#4a5c91;margin-bottom:8px;display:block;}"
                ".prompt-text{white-space:pre-wrap;font-family:monospace;font-size:13px;line-height:1.5;color:#1f2d4d;}"
                ".response-stream{white-space:pre-wrap;font-family:monospace;font-size:13px;line-height:1.6;background:#fff;border:1px solid #e3e8f8;border-radius:8px;padding:16px;color:#19213b;}"
                ".response-token{display:inline;white-space:pre;border-radius:4px;padding:2px 3px;margin:0 1px;transition:box-shadow 0.1s ease,transform 0.1s ease;color:inherit;position:relative;}"
                ".response-token:hover{box-shadow:0 4px 12px rgba(31,37,48,0.18);transform:translateY(-1px);z-index:2;}"
                ".response-token.whitespace{margin:0;}"
                ".response-token .tooltip{position:absolute;left:0;bottom:100%;transform:translateY(-6px);background:#1f2530;color:#f8f9ff;padding:8px 10px;border-radius:6px;font-size:12px;line-height:1.45;white-space:normal;box-shadow:0 8px 18px rgba(15,23,42,0.3);opacity:0;visibility:hidden;pointer-events:none;min-width:220px;max-width:320px;}"
                ".response-token:hover .tooltip{opacity:1;visibility:visible;}"
                ".response-token .tooltip::after{content:\"\";position:absolute;top:100%;left:12px;border-width:6px;border-style:solid;border-color:#1f2530 transparent transparent transparent;}"
                ".tooltip-entropy{font-weight:600;margin-bottom:4px;color:#fefefe;}"
                ".tooltip-logprob{font-family:monospace;font-size:12px;margin-bottom:6px;color:rgba(248,249,255,0.9);}"
                ".tooltip-logprobs-title{font-size:11px;text-transform:uppercase;letter-spacing:0.05em;color:rgba(248,249,255,0.7);margin-bottom:4px;}"
                ".tooltip-logprobs{display:flex;flex-direction:column;gap:2px;}"
                ".tooltip-logprob-row{display:flex;justify-content:space-between;gap:12px;font-family:monospace;font-size:12px;color:#fefefe;}"
                ".tooltip-empty{font-style:italic;color:rgba(248,249,255,0.7);}"
                "</style></head><body>"
            )
            handle.write("<h1>Token Entropy Report</h1>")

            if observations:
                anchor_counts: Dict[str, int] = {}
                nav_items: List[tuple[str, Observation]] = []
                for idx, observation in enumerate(observations):
                    base_anchor = sanitize_id(observation.record_id, f"record-{idx}")
                    occurrence = anchor_counts.get(base_anchor, 0)
                    anchor_counts[base_anchor] = occurrence + 1
                    safe_anchor = (
                        base_anchor if occurrence == 0 else f"{base_anchor}-{occurrence}"
                    )
                    nav_items.append((safe_anchor, observation))

            if not observations:
                handle.write("<p>No observations to display.</p>")
                handle.write("</body></html>")
                return

            for safe_anchor, observation in nav_items:
                entropies: List[Optional[float]] = observation.metrics.get("token_entropies") or []
                max_entropy = observation.metrics.get("max_token_entropy")

                display_tokens: List[Dict[str, Any]] = []
                row_index = 0
                for idx, token in enumerate(observation.tokens):
                    if token.is_special:
                        continue
                    row_index += 1
                    entropy_value = entropies[idx] if idx < len(entropies) else None
                    if entropy_value is not None and abs(entropy_value) < 1e-9:
                        entropy_value = 0.0
                    top_logprobs = token.top_logprobs or {}
                    if max_entropy and max_entropy > 0 and entropy_value is not None:
                        normalized = max(0.0, min(1.0, entropy_value / max_entropy))
                    else:
                        normalized = 0.0
                    highlight_alpha = 0.20 + (0.70 * normalized)
                    highlight_alpha = max(0.0, min(0.95, highlight_alpha))
                    entropy_attr = (
                        f"{entropy_value:.4f}" if entropy_value is not None else "nan"
                    )
                    if entropy_value is not None:
                        chip_style = f"background-color: rgba(255, 99, 71, {highlight_alpha:.3f});color:#241517;"
                    else:
                        chip_style = "background-color: #e9ecf5;"
                    logprob_value = token.logprob
                    top_items = list(top_logprobs.items())
                    tooltip_rows_html = "".join(
                        f"<div class='tooltip-logprob-row'><span class='candidate'>{format_token_text(tok)}</span>"
                        f"<span class='score'>{value:.4f}</span></div>"
                        for tok, value in top_items
                    )
                    if entropy_value is not None:
                        entropy_display = f"{entropy_value:.4f}"
                    else:
                        entropy_display = "N/A"
                    if logprob_value is not None:
                        logprob_display = (
                            f"{0.0:.4f}" if abs(logprob_value) < 1e-9 else f"{logprob_value:.4f}"
                        )
                    else:
                        logprob_display = "N/A"
                    tooltip_logprobs_block = (
                        "<div class='tooltip-logprobs-title'>Top logprobs</div>"
                        f"<div class='tooltip-logprobs'>{tooltip_rows_html}</div>"
                        if tooltip_rows_html
                        else "<div class='tooltip-logprobs-title'>Top logprobs</div><div class='tooltip-empty'>No data</div>"
                    )
                    tooltip_html = (
                        "<div class='tooltip'>"
                        f"<div class='tooltip-entropy'>Entropy: {entropy_display}</div>"
                        f"<div class='tooltip-logprob'>Logprob: {logprob_display}</div>"
                        f"{tooltip_logprobs_block}"
                        "</div>"
                    )
                    title_parts = [
                        f"Token #{row_index} (orig {idx})",
                        f"logprob: {logprob_display}",
                        f"entropy: {entropy_display}",
                    ]
                    title_attr = " | ".join(title_parts)
                    display_tokens.append(
                        {
                            "response_text": escape(token.text),
                            "entropy_attr": entropy_attr,
                            "is_whitespace": token.text.strip() == "",
                            "chip_style": chip_style,
                            "tooltip_html": tooltip_html,
                            "title_attr": title_attr,
                        }
                    )

                handle.write(f"<section id='{safe_anchor}' class='response-section'>")
                handle.write(
                    f"<h2 class='record-title'>Record {escape(observation.record_id)}</h2>"
                )
                prompt_text = (
                    observation.metadata.get("prompt")
                    or observation.metadata.get("record_metadata", {}).get("prompt")
                    or observation.metadata.get("record_metadata", {}).get("metadata", {}).get("prompt")
                )
                if prompt_text:
                    handle.write("<div class='prompt-block'>")
                    handle.write("<span class='prompt-label'>Prompt</span>")
                    handle.write(f"<div class='prompt-text'>{escape(str(prompt_text))}</div>")
                    handle.write("</div>")

                handle.write("<div class='response-stream'>")
                for token_info in display_tokens:
                    response_classes = ["response-token"]
                    if token_info["is_whitespace"]:
                        response_classes.append("whitespace")
                    handle.write(
                        f"<span class='{' '.join(response_classes)}' data-entropy='{token_info['entropy_attr']}' "
                        f"style='{token_info['chip_style']}' title='{escape(token_info['title_attr'])}'>"
                        f"{token_info['response_text']}{token_info['tooltip_html']}</span>"
                    )
                handle.write("</div>")

                handle.write("</section>")

            handle.write("</body></html>")
