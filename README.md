# LLM Observer Framework

This repository contains a lightweight framework for sending code-oriented prompts to an LLM and recording a suite of observation metrics—including per-token entropy—to help you understand response quality in real time.

## Quick Start

1. Create a JSONL dataset containing the prompts you want to evaluate (`datasets/example.jsonl` shows the expected format).
2. Ensure the `OPENAI_API_KEY` environment variable is set if you intend to call OpenAI models.
3. Run the observer:

```bash
python run_observer.py --dataset datasets/example.jsonl --output logs/observations.jsonl --include-tokens
```

Add `--dry-run` to skip remote calls while testing wiring; the script will echo the prompt back.

## Metrics Captured

- `latency_ms`: Wall-clock latency per request.
- `response_char_length` / `response_token_count` / `response_line_count`: Size of the output.
- `placeholder_count` / `has_placeholders`: Heuristic check for incomplete answers.
- `mean_token_entropy` / `max_token_entropy`: Calculated from the per-token log-probability distribution (requires API support for logprobs).
- `token_entropies`: Raw entropy sequence (included when `--include-tokens` is set).
- `syntax_valid`: Python syntax check (extendable for other languages).
- `exact_match`: Simple string comparison against `reference` ground truth if provided.

You can attach additional evaluators or metrics by supplying callables to `LLMObserverRunner`.

## Extending

- Implement custom evaluators by providing a function with signature `(generation, record) -> dict`.
- Swap in different `LLMClient` implementations for other providers while preserving the `LLMGeneration` contract.
- Extend `compute_syntax_validity` to add language-specific compilers or linters.

## Outputs

The runner writes each observation as a JSON object containing the response text, collected metrics, and optional metadata (prompt, tokens, raw payload) depending on CLI flags. Use the `logs/observations.jsonl` file to build dashboards or further analytics.
