#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_observer.llm_client import LLMToken
from llm_observer.runner import LLMObserverRunner, Observation, ObserverConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute token-level entropy for prompt/response pairs using a local HuggingFace model.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a JSONL file with fields: id, prompt, response.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path or identifier of the HuggingFace causal language model.",
    )
    parser.add_argument(
        "--html-report",
        help="Optional path to write an HTML entropy report.",
    )
    parser.add_argument(
        "--json-output",
        help="Optional path to write recomputed observations as JSONL.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of alternatives to keep for top-logprobs (default: 5).",
    )
    return parser.parse_args()


def load_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            if "prompt" not in payload or "response" not in payload:
                raise ValueError(
                    f"Line {line_number} missing required 'prompt' or 'response' fields."
                )
            payload.setdefault("id", f"record-{len(records)}")
            records.append(payload)
    if not records:
        raise ValueError("Input file is empty.")
    return records


def load_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() else None
    load_kwargs: Dict[str, Any] = {"torch_dtype": dtype}
    if device_map:
        load_kwargs["device_map"] = device_map

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    if device_map is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    model.eval()
    return tokenizer, model


def _gather_top_logprobs(
    log_probs: torch.Tensor,
    tokenizer,
    top_k: int,
) -> Dict[str, float]:
    values, indices = torch.topk(log_probs, k=min(top_k, log_probs.size(-1)))
    top: Dict[str, float] = {}
    for idx, val in zip(indices.tolist(), values.tolist()):
        if math.isinf(val) and val < 0.0:
            continue
        token_text = tokenizer.convert_ids_to_tokens([idx])[0]
        top[token_text] = float(val)
    return top


def recompute_record(
    tokenizer,
    model,
    record: Dict[str, Any],
    *,
    top_k: int,
) -> Tuple[List[LLMToken], List[Optional[float]], Dict[str, Any]]:
    prompt = record["prompt"]
    response = record["response"]

    prompt_enc = tokenizer(prompt, return_tensors="pt")
    response_enc = tokenizer(response, return_tensors="pt", add_special_tokens=False)

    input_ids = torch.cat([prompt_enc["input_ids"], response_enc["input_ids"]], dim=1)
    attention_mask = torch.cat(
        [prompt_enc["attention_mask"], response_enc["attention_mask"]], dim=1
    )

    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs.logits  # (1, seq_len, vocab)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    target_ids = input_ids[:, 1:]
    log_probs = log_probs[:, :-1, :]

    token_logprobs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
    entropies = -(torch.exp(log_probs) * log_probs).sum(dim=-1)

    prompt_len = prompt_enc["input_ids"].size(1)
    response_len = response_enc["input_ids"].size(1)
    if response_len == 0:
        return [], [], {
            "mean_token_entropy": None,
            "max_token_entropy": None,
            "token_entropies": [],
        }

    prediction_start = prompt_len - 1
    prediction_end = prediction_start + response_len

    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])

    tokens: List[LLMToken] = []
    per_token_entropies: List[Optional[float]] = []
    numeric_entropies: List[float] = []

    for position in range(prediction_start, prediction_end):
        token_id = target_ids[0, position].item()
        token_text = tokenizer.convert_ids_to_tokens([token_id])[0]
        logprob = float(token_logprobs[0, position].item())
        entropy_value = float(entropies[0, position].item())
        top_logprobs = _gather_top_logprobs(log_probs[0, position], tokenizer, top_k)

        is_special = token_id in special_ids
        if is_special:
            per_token_entropies.append(None)
        else:
            per_token_entropies.append(entropy_value)
            numeric_entropies.append(entropy_value)

        tokens.append(
            LLMToken(
                text=token_text,
                logprob=logprob,
                top_logprobs=top_logprobs,
                is_special=is_special,
            )
        )

    metrics = {
        "mean_token_entropy": (
            sum(numeric_entropies) / len(numeric_entropies) if numeric_entropies else None
        ),
        "max_token_entropy": max(numeric_entropies) if numeric_entropies else None,
        "token_entropies": per_token_entropies,
    }

    return tokens, per_token_entropies, metrics


def build_observations(
    records: Sequence[Dict[str, Any]],
    tokenizer,
    model,
    *,
    top_k: int,
) -> List[Observation]:
    observations: List[Observation] = []

    for record in records:
        tokens, per_token_entropies, metrics = recompute_record(
            tokenizer,
            model,
            record,
            top_k=top_k,
        )

        observation = Observation(
            record_id=str(record.get("id", "")),
            metrics=metrics,
            response_text=record["response"],
            tokens=tokens,
            metadata={
                "record_metadata": {
                    key: record[key]
                    for key in record
                    if key not in {"response"}
                },
                "prompt": record["prompt"],
            },
        )
        observations.append(observation)

    return observations


def write_jsonl(path: Path, observations: Sequence[Observation]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for obs in observations:
            payload = {
                "id": obs.record_id,
                "response": obs.response_text,
                "metrics": obs.metrics,
                "metadata": obs.metadata,
                "tokens": [
                    {
                        "text": token.text,
                        "logprob": token.logprob,
                        "top_logprobs": token.top_logprobs,
                        "is_special": token.is_special,
                    }
                    for token in obs.tokens
                ],
            }
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    records = load_records(input_path)

    tokenizer, model = load_model(args.model)

    observations = build_observations(
        records,
        tokenizer,
        model,
        top_k=args.top_k,
    )

    for obs in observations:
        mean_entropy = obs.metrics.get("mean_token_entropy")
        max_entropy = obs.metrics.get("max_token_entropy")
        print(f"[{obs.record_id}] mean entropy: {mean_entropy}, max entropy: {max_entropy}")

    if args.json_output:
        write_jsonl(Path(args.json_output), observations)

    if args.html_report:
        dummy_client = object()
        config = ObserverConfig(html_report_path=Path(args.html_report))
        runner = LLMObserverRunner(dummy_client, config=config)
        runner._write_html_report(list(observations))


if __name__ == "__main__":
    main()
