#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Any, Dict, List

from llm_observer.dataset import load_jsonl_dataset
from llm_observer.llm_client import EchoClient, LocalHFClient, OpenAIClient
from llm_observer.runner import LLMObserverRunner, ObserverConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a dataset of prompts through an LLM and collect observation metrics.",
    )
    parser.add_argument("--dataset", required=True, help="Path to the JSONL dataset.")
    parser.add_argument("--output", help="Optional path to write observation logs (JSONL).")
    parser.add_argument(
        "--html-report",
        help="Optional path to write an HTML entropy report for each observation.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="Model identifier for the OpenAI client.",
    )
    parser.add_argument(
        "--use-chat",
        action="store_true",
        help="Use the Chat Completions API instead of the Responses API.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Optional temperature override for the LLM request."
    )
    parser.add_argument(
        "--max-output-tokens", type=int, help="Optional max output tokens for the LLM request."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use an echo client instead of calling the OpenAI API (useful for testing).",
    )
    parser.add_argument(
        "--include-prompt",
        action="store_true",
        help="Store the original prompt in the metadata output.",
    )
    parser.add_argument(
        "--include-tokens",
        action="store_true",
        help="Store token-level logprob data in the metadata output.",
    )
    parser.add_argument(
        "--include-raw",
        action="store_true",
        help="Store raw LLM responses in the metadata output.",
    )
    return parser.parse_args()


def build_client(args: argparse.Namespace):
    if args.dry_run:
        return EchoClient()

    request_kwargs: Dict[str, Any] = {}
    if args.temperature is not None:
        request_kwargs["temperature"] = args.temperature
    if args.max_output_tokens is not None:
        request_kwargs["max_output_tokens"] = args.max_output_tokens

    model_path = Path(args.model)
    if model_path.exists():
        return LocalHFClient(
            model_path=str(model_path),
            generation_kwargs=request_kwargs,
        )

    return OpenAIClient(
        model=args.model,
        request_kwargs=request_kwargs,
        use_responses_api=not args.use_chat,
    )


def build_config(args: argparse.Namespace) -> ObserverConfig:
    save_path = Path(args.output) if args.output else None
    html_report_path = Path(args.html_report) if args.html_report else None
    extra_request_args: Dict[str, Any] = {}
    if args.temperature is not None:
        extra_request_args["temperature"] = args.temperature
    if args.max_output_tokens is not None:
        extra_request_args["max_output_tokens"] = args.max_output_tokens

    return ObserverConfig(
        save_path=save_path,
        include_prompt=args.include_prompt,
        include_tokens=args.include_tokens,
        include_raw_response=args.include_raw,
        extra_request_args=extra_request_args,
        html_report_path=html_report_path,
    )


def summarize(observations) -> None:
    count = len(observations)
    latencies = [
        obs.metrics.get("latency_ms")
        for obs in observations
        if obs.metrics.get("latency_ms") is not None
    ]
    entropies = [
        obs.metrics.get("mean_token_entropy")
        for obs in observations
        if obs.metrics.get("mean_token_entropy") is not None
    ]
    exact_matches = [
        obs.metrics.get("exact_match")
        for obs in observations
        if obs.metrics.get("exact_match") is not None
    ]

    print(f"Processed {count} records.")
    if latencies:
        print(f"  Median latency: {statistics.median(latencies):.1f} ms")
        print(f"  Mean latency: {statistics.mean(latencies):.1f} ms")
    if entropies:
        print(f"  Mean of mean token entropy: {statistics.mean(entropies):.3f}")
    if exact_matches:
        success_rate = sum(1 for match in exact_matches if match) / len(exact_matches)
        print(f"  Exact-match success rate: {success_rate:.1%}")


def main() -> None:
    args = parse_args()
    dataset = load_jsonl_dataset(args.dataset)
    if not dataset:
        raise SystemExit("Dataset is empty.")

    client = build_client(args)
    config = build_config(args)

    runner = LLMObserverRunner(client, config=config)
    observations = runner.run(dataset)

    summarize(observations)


if __name__ == "__main__":
    main()
