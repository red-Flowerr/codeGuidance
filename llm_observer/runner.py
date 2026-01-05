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
                "h1{margin-top:0;font-size:28px;color:#182140;}"
                "h2{margin-top:48px;font-size:22px;color:#273352;}"
                "h3{margin-top:24px;font-size:18px;color:#2f3b62;}"
                "nav{position:sticky;top:0;padding:12px 16px;margin:-24px -24px 24px;background:#fff;border-bottom:1px solid #d8deeb;display:flex;gap:16px;align-items:center;z-index:1;}"
                "nav strong{font-size:14px;text-transform:uppercase;letter-spacing:0.08em;color:#5c6786;}"
                "nav ul{list-style:none;margin:0;padding:0;display:flex;flex-wrap:wrap;gap:12px;}"
                "nav a{text-decoration:none;color:#2a4ac5;background:#e7ecff;padding:6px 10px;border-radius:6px;font-size:13px;font-family:monospace;}"
                "nav a:hover{background:#d4defd;}"
                "section{background:#fff;border:1px solid #e1e6f3;box-shadow:0 8px 16px rgba(15,23,42,0.06);border-radius:10px;padding:24px;margin-bottom:32px;}"
                ".summary{list-style:none;margin:16px 0;padding:0;display:flex;flex-wrap:wrap;gap:16px;background:#f7f8fe;border:1px solid #e3e8f8;border-radius:8px;padding:16px;}"
                ".summary li{min-width:200px;font-family:monospace;font-size:14px;color:#3a4663;}"
                "pre{background:#0d1117;color:#e6edf3;padding:16px;border-radius:8px;overflow:auto;font-size:13px;line-height:1.55;}"
                "table{border-collapse:collapse;width:100%;margin-top:18px;font-family:monospace;font-size:13px;}"
                "thead th{position:sticky;top:48px;background:#eef2ff;z-index:1;}"
                "th,td{border:1px solid #d5dcf0;padding:6px 8px;text-align:left;vertical-align:top;}"
                "tbody tr:nth-child(odd){background:#fbfcff;}"
                "tbody tr:hover{background:#f1f4ff;}"
                ".token{white-space:pre;font-weight:600;color:#1c2543;}"
                ".entropy-cell{position:relative;}"
                ".entropy-cell[data-entropy='nan']{background:#f7f8fe;color:#7b849d;}"
                ".top-logprobs{min-width:220px;}"
                ".top-logprobs div{display:flex;justify-content:space-between;gap:12px;padding:2px 0;border-bottom:1px dotted #dde3f8;}"
                ".top-logprobs div:last-child{border-bottom:none;}"
                ".candidate{color:#1f2d5c;}"
                ".score{color:#5c6786;}"
                ".token-index{color:#607099;font-weight:500;}"
                ".token-stream{display:flex;flex-wrap:wrap;gap:6px;padding:12px;border:1px solid #e3e8f8;background:#f9faff;border-radius:8px;margin-top:12px;}"
                ".token-chip{display:inline-flex;align-items:center;padding:4px 8px;border-radius:6px;font-family:monospace;font-size:12px;color:#19213b;white-space:pre;box-shadow:0 1px 2px rgba(31,37,48,0.12);border:1px solid rgba(52,82,145,0.08);transition:transform 0.1s ease, box-shadow 0.1s ease;}"
                ".token-chip:hover{box-shadow:0 4px 10px rgba(31,37,48,0.18);transform:translateY(-1px);}"
                ".token-chip.whitespace{color:#6b738a;background:#eef1fa;border-style:dashed;}"
                ".token-chip.high{border-color:#f46f43;box-shadow:0 2px 8px rgba(244,111,67,0.3);}"
                ".token-chip.hidden{display:none;}"
                ".controls{display:flex;flex-wrap:wrap;align-items:center;gap:16px;padding:12px 16px;border:1px solid #dfe4f6;background:#f7f8fe;border-radius:8px;margin-bottom:20px;font-family:system-ui,-apple-system,\"Segoe UI\",sans-serif;}"
                ".controls label{font-size:13px;color:#2d3c66;display:flex;align-items:center;gap:8px;}"
                ".controls input[type=\"range\"]{width:160px;}"
                ".legend{display:flex;align-items:center;gap:12px;font-size:12px;color:#4c5675;margin-bottom:20px;font-family:system-ui,-apple-system,\"Segoe UI\",sans-serif;}"
                ".legend-bar{position:relative;flex:1;height:10px;border-radius:6px;background:linear-gradient(90deg,rgba(244,111,67,0.15) 0%,rgba(244,111,67,0.7) 100%);}"
                ".legend-bar::before,.legend-bar::after{content:\"\";position:absolute;top:50%;transform:translateY(-50%);width:1px;height:14px;background:#c8cee6;}"
                ".legend-bar::before{left:0;}"
                ".legend-bar::after{right:0;}"
                ".legend span[data-label]{font-size:12px;color:#5d6687;}"
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
                "details{margin-top:18px;}"
                "details summary{cursor:pointer;font-weight:600;color:#2a3a68;margin-bottom:12px;}"
                "details[open] summary{margin-bottom:12px;}"
                "@media (max-width:960px){nav{flex-direction:column;align-items:flex-start;}thead th{top:0;}}"
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

                handle.write("<nav><strong>Records</strong><ul>")
                for safe_anchor, observation in nav_items:
                    handle.write(
                        f"<li><a href='#{safe_anchor}'>{escape(observation.record_id)}</a></li>"
                    )
                handle.write("</ul></nav>")

            if not observations:
                handle.write("<p>No observations to display.</p>")
                handle.write("</body></html>")
                return

            handle.write(
                "<div class='controls'>"
                "<label>Highlight entropy ≥ <span id='thresholdValue'>0.50</span></label>"
                "<input type='range' id='thresholdSlider' min='0' max='3.0' step='0.05' value='0.50'>"
                "<label><input type='checkbox' id='hideLowEntropy'> Hide lower-entropy tokens</label>"
                "</div>"
            )
            handle.write(
                "<div class='legend'><span data-label>Low</span>"
                "<div class='legend-bar'></div><span data-label>High</span></div>"
            )

            for safe_anchor, observation in nav_items:
                entropies: List[Optional[float]] = observation.metrics.get("token_entropies") or []
                mean_entropy = observation.metrics.get("mean_token_entropy")
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
                    highlight_alpha = 0.15 + (0.55 * normalized)
                    highlight_alpha = max(0.0, min(0.7, highlight_alpha))
                    entropy_attr = (
                        f"{entropy_value:.4f}" if entropy_value is not None else "nan"
                    )
                    entropy_style = (
                        f"background-color: rgba(244, 111, 67, {highlight_alpha:.3f});"
                        if entropy_value is not None
                        else ""
                    )
                    chip_style = entropy_style or "background-color: #e9ecf5;"
                    logprob_value = token.logprob
                    top_items = list(top_logprobs.items())
                    top_items_html = "".join(
                        f"<div><span class='candidate'>{format_token_text(tok)}</span>"
                        f"<span class='score'>{value:.4f}</span></div>"
                        for tok, value in top_items
                    )
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
                            "row_index": row_index,
                            "original_index": idx,
                            "token_text": token.text,
                            "chip_text": format_token_text(token.text),
                            "response_text": escape(token.text),
                            "logprob": logprob_value,
                            "entropy": entropy_value,
                            "entropy_numeric": entropy_value,
                            "entropy_attr": entropy_attr,
                            "entropy_style": entropy_style,
                            "top_items_html": top_items_html,
                            "is_whitespace": token.text.strip() == "",
                            "chip_style": chip_style,
                            "chip_attr": entropy_attr,
                            "tooltip_html": tooltip_html,
                            "title_attr": title_attr,
                            "logprob_display": logprob_display,
                        }
                    )

                handle.write(f"<section id='{safe_anchor}'>")
                handle.write(f"<h2>Record {escape(observation.record_id)}</h2>")
                handle.write("<ul class='summary'>")
                if mean_entropy is not None:
                    handle.write(f"<li>Mean entropy: {mean_entropy:.4f}</li>")
                if max_entropy is not None:
                    handle.write(f"<li>Max entropy: {max_entropy:.4f}</li>")
                handle.write(f"<li>Response length (tokens): {len(display_tokens)}</li>")
                handle.write("</ul>")

                handle.write("<h3>Generated Response</h3>")
                handle.write("<div class='response-stream'>")
                for token_info in display_tokens:
                    response_classes = ["response-token"]
                    if token_info["is_whitespace"]:
                        response_classes.append("whitespace")
                    handle.write(
                        f"<span class='{' '.join(response_classes)}' data-entropy='{token_info['chip_attr']}' "
                        f"style='{token_info['chip_style']}' title='{escape(token_info['title_attr'])}'>"
                        f"{token_info['response_text']}{token_info['tooltip_html']}</span>"
                    )
                handle.write("</div>")

                handle.write("<h3>Entropy Overview</h3>")
                handle.write("<div class='token-stream'>")
                for token_info in display_tokens:
                    classes = ["token-chip"]
                    if token_info["is_whitespace"]:
                        classes.append("whitespace")
                    handle.write(
                        f"<span class='{' '.join(classes)}' data-entropy='{token_info['chip_attr']}' "
                        f"style='{token_info['chip_style']}' title='{escape(token_info['title_attr'])}'>"
                        f"{token_info['chip_text']}</span>"
                    )
                handle.write("</div>")

                handle.write("<h3>Token Details</h3>")
                handle.write("<details open>")
                handle.write("<summary>Detailed token metrics</summary>")
                handle.write(
                    "<table><thead><tr><th>#</th><th>Token</th><th>Logprob</th>"
                    "<th>Entropy</th><th>Top Logprobs</th></tr></thead><tbody>"
                )
                for token_info in display_tokens:
                    entropy_attr = token_info["entropy_attr"]
                    entropy_style = token_info["entropy_style"]
                    entropy_value = token_info["entropy_numeric"]
                    handle.write("<tr>")
                    handle.write(
                        f"<td class='token-index' title='Original index {token_info['original_index']}'>"
                        f"{token_info['row_index']}</td>"
                    )
                    token_classes = "token whitespace" if token_info["is_whitespace"] else "token"
                    handle.write(f"<td class='{token_classes}'>{token_info['chip_text']}</td>")
                    handle.write(f"<td>{token_info['logprob_display']}</td>")
                    if entropy_value is not None:
                        entropy_display = f"{entropy_value:.4f}"
                        handle.write(
                            f"<td class='entropy-cell' data-entropy='{entropy_attr}' style='{entropy_style}'>"
                            f"{entropy_display}</td>"
                        )
                    else:
                        handle.write("<td class='entropy-cell' data-entropy='nan'>N/A</td>")
                    handle.write(f"<td class='top-logprobs'>{token_info['top_items_html']}</td>")
                    handle.write("</tr>")
                handle.write("</tbody></table>")
                handle.write("</details>")

                handle.write("</section>")

            handle.write(
                "<script>"
                "const slider=document.getElementById('thresholdSlider');"
                "const sliderValue=document.getElementById('thresholdValue');"
                "const hideLow=document.getElementById('hideLowEntropy');"
                "const chips=document.querySelectorAll('.token-chip');"
                "function applyThreshold(){"
                "const threshold=parseFloat(slider.value);"
                "sliderValue.textContent=threshold.toFixed(2);"
                "chips.forEach(chip=>{"
                "const val=parseFloat(chip.dataset.entropy);"
                "if(Number.isNaN(val)){chip.classList.remove('high');chip.classList.remove('hidden');return;}"
                "if(val>=threshold){chip.classList.add('high');chip.classList.remove('hidden');}"
                "else{chip.classList.remove('high');if(hideLow.checked){chip.classList.add('hidden');}else{chip.classList.remove('hidden');}}"
                "});"
                "}"
                "slider.addEventListener('input',applyThreshold);"
                "hideLow.addEventListener('change',applyThreshold);"
                "applyThreshold();"
                "</script>"
            )

            handle.write("</body></html>")
