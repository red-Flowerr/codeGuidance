#!/usr/bin/env python3
"""
Utility script to run Qwen2.5-Coder on a single LiveCodeBench sample and
print token-level log probabilities (logits) together with the generated code.
"""

import argparse
import html
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_qwen_prompt(sample: Dict[str, Any]) -> str:
    """Construct the Qwen chat-style prompt expected by CodeQwen instruct models."""
    system = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user"
    guidance = (
        "You will be given a question (problem specification) and will generate a correct "
        "Python program that matches the specification and passes all tests. "
        "You will NOT return anything except for the program.\n\n"
    )

    question = sample.get("question_content", "")
    starter_code = sample.get("starter_code") or ""
    prompt = f"{system}\n\n{guidance}Question: {question}\n\n"

    if starter_code.strip():
        prompt += (
            "You will use the following starter code to write the solution to the problem "
            "and enclose your code within delimiters.\n"
        )
        prompt += f"```python\n{starter_code}\n```\n\n<|im_end|>\n"
    else:
        prompt += (
            "Read the inputs from stdin solve the problem and write the answer to stdout "
            "(do not directly test on the sample inputs). Enclose your code within "
            "delimiters as follows. Ensure that when the python program runs, it reads the "
            "inputs, runs the algorithm and writes output to STDOUT.\n"
        )
        prompt += "```python\n# YOUR CODE HERE\n```\n\n<|im_end|>\n"

    prompt += "<|im_start|>assistant\n"
    return prompt


def load_sample(path: Path, index: int) -> Dict[str, Any]:
    """Load one sample (dict) from a JSON array file."""
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array.")
    if index < 0 or index >= len(data):
        raise IndexError(f"Index {index} out of range for file with {len(data)} entries.")
    return data[index]


def collect_token_stats(
    tokenizer: AutoTokenizer,
    generated_ids: torch.Tensor,
    scores: List[torch.Tensor],
) -> List[Dict[str, Any]]:
    """Match every generated token with its logprob and top-K alternatives."""
    # scores is a list of (batch, vocab_size) tensors; stack and move to CPU
    stacked = torch.stack(scores).float().cpu().squeeze(1)
    new_tokens = generated_ids.cpu()
    logprobs = torch.nn.functional.log_softmax(stacked, dim=-1)

    topk_logprobs, topk_indices = torch.topk(logprobs, k=5, dim=-1)
    probs = logprobs.exp()
    entropies = (-probs * logprobs).sum(dim=-1)

    records: List[Dict[str, Any]] = []
    token_texts = tokenizer.convert_ids_to_tokens(new_tokens.tolist())
    for idx, token_id in enumerate(new_tokens):
        selected_lp = logprobs[idx, token_id].item()
        top_entries = [
            {
                "token": tokenizer.convert_ids_to_tokens([topk_indices[idx, j].item()])[0],
                "logprob": topk_logprobs[idx, j].item(),
            }
            for j in range(topk_indices.size(1))
        ]
        records.append(
            {
                "index": idx,
                "token": token_texts[idx],
                "logprob": selected_lp,
                "entropy": entropies[idx].item(),
                "top_alternatives": top_entries,
            }
        )
    return records


def run_generation(args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    sample = load_sample(Path(args.sample_file), args.record_index)
    prompt = build_qwen_prompt(sample)

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    start = time.perf_counter()
    with torch.no_grad():
        generate_kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0.0,
            return_dict_in_generate=True,
            output_scores=True,
        )
        if args.temperature > 0.0:
            generate_kwargs["temperature"] = args.temperature
        outputs = model.generate(**inputs, **generate_kwargs)
    latency_ms = (time.perf_counter() - start) * 1000.0

    sequence = outputs.sequences[0]
    input_len = inputs["input_ids"].shape[1]
    new_tokens = sequence[input_len:]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=False)

    token_records = collect_token_stats(tokenizer, new_tokens, outputs.scores)
    token_entropies = [item["entropy"] for item in token_records]

    return {
        "latency_ms": latency_ms,
        "prompt_tokens": input_len,
        "response_tokens": len(new_tokens),
        "response_text": response_text,
        "token_logprobs": token_records,
        "mean_token_entropy": sum(token_entropies) / len(token_entropies) if token_entropies else None,
        "max_token_entropy": max(token_entropies) if token_entropies else None,
    }


def save_metric_plot(
    token_records: List[Dict[str, Any]],
    output_path: Path,
    limit: Optional[int] = None,
    metric: str = "logprob",
) -> None:
    """Render a simple line plot of a token-level metric (logprob or entropy)."""
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:  # pragma: no cover - visualization helper
        raise RuntimeError(
            "matplotlib is required for --plot-path. Install it with 'pip install matplotlib'."
        ) from exc

    subset = token_records[:limit] if limit else token_records
    if not subset:
        raise ValueError("No token records available to plot.")

    indices = [item["index"] for item in subset]
    if metric == "entropy":
        values = [item["entropy"] for item in subset]
        ylabel = "Token entropy (nats)"
        title = "Qwen2.5-Coder Token Entropy"
    else:
        metric = "logprob"
        values = [item["logprob"] for item in subset]
        ylabel = "Log probability (natural log)"
        title = "Qwen2.5-Coder Token Logprobs"

    tokens = [
        item["token"]
        .replace("Ġ", "▁")
        .replace("Ċ", "\\n")
        .replace("Ħ", "#")
        for item in subset
    ]

    fig_width = max(6.0, min(20.0, len(subset) * 0.15))
    fig, ax = plt.subplots(figsize=(fig_width, 4.5))
    ax.plot(indices, values, marker="o", linewidth=1.0, markersize=3.0, color="#1f77b4")
    ax.set_xlabel("Token index")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, linestyle="--")

    if len(subset) <= 60:
        ax.set_xticks(indices)
        ax.set_xticklabels(tokens, rotation=90, fontsize=8)
    else:
        step = max(1, len(subset) // 20)
        tick_indices = indices[::step]
        tick_labels = tokens[::step]
        ax.set_xticks(tick_indices)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate code with Qwen2.5-Coder and dump per-token logprobs."
    )
    parser.add_argument(
        "--model-path",
        default="/mnt/hdfs/tiktok_aiic/user/codeai/hf_models/Qwen2.5-Coder-7B-Instruct",
        help="Path to the Hugging Face model directory.",
    )
    parser.add_argument(
        "--sample-file",
        default="LiveCodeBench/output/oc_only_2epoch/Scenario.codegeneration_10_0.2.json",
        help="JSON file containing an array of LiveCodeBench records.",
    )
    parser.add_argument(
        "--record-index",
        type=int,
        default=0,
        help="Which record in the JSON file to evaluate.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 keeps greedy decoding).",
    )
    parser.add_argument(
        "--output-json",
        help="Optional path to write the full JSON result. Defaults to stdout only.",
    )
    parser.add_argument(
        "--plot-path",
        help="Optional path to save a PNG plot of token logprobs.",
    )
    parser.add_argument(
        "--plot-limit",
        type=int,
        default=200,
        help="Maximum number of tokens to include in the plot (default: 200).",
    )
    parser.add_argument(
        "--plot-metric",
        choices=["logprob", "entropy"],
        default="logprob",
        help="Metric to plot when --plot-path is set (default: logprob).",
    )
    parser.add_argument(
        "--html-path",
        help="Optional path to write an interactive HTML report (response + metric chart).",
    )
    return parser.parse_args()


def save_html_report(
    token_records: List[Dict[str, Any]],
    response_text: str,
    output_path: Path,
) -> None:
    """Create a simple HTML report linking the response text to token metrics."""
    if not token_records:
        raise ValueError("No token records available for HTML report.")

    response_html = html.escape(response_text)
    indices = [rec["index"] for rec in token_records]
    tokens_readable = [
        rec["token"]
        .replace("Ġ", "▁")
        .replace("Ċ", "\\n")
        .replace("Ħ", "#")
        for rec in token_records
    ]
    entropies = [rec["entropy"] for rec in token_records]
    logprobs = [rec["logprob"] for rec in token_records]

    js_payload = {
        "indices": indices,
        "tokens": tokens_readable,
        "entropies": entropies,
        "logprobs": logprobs,
    }

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>Qwen Token Metrics</title>
    <style>
      body {{
        font-family: Arial, sans-serif;
        margin: 20px;
      }}
      h1 {{
        font-size: 20px;
        margin-bottom: 10px;
      }}
      pre {{
        background: #f5f5f5;
        padding: 12px;
        border-radius: 6px;
        white-space: pre-wrap;
      }}
      #token-strip {{
        margin-top: 20px;
        font-family: "Courier New", monospace;
        display: flex;
        flex-wrap: wrap;
        gap: 4px;
      }}
      .token {{
        padding: 2px 4px;
        border-radius: 4px;
        background-color: #e8eef6;
        cursor: pointer;
      }}
      .token.active {{
        background-color: #1f77b4;
        color: white;
      }}
      #metrics {{
        margin-top: 14px;
        font-size: 14px;
      }}
      canvas {{
        max-width: 100%;
      }}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  </head>
  <body>
    <h1>Model Response</h1>
    <pre id="response">{response_html}</pre>

    <h2>Token Metrics</h2>
    <div id="metrics">Hover a token below to view logprob & entropy.</div>
    <div id="token-strip"></div>
    <canvas id="metricChart" height="220"></canvas>

    <script>
      const data = {json.dumps(js_payload, ensure_ascii=False)};
      const strip = document.getElementById("token-strip");
      const metrics = document.getElementById("metrics");

      let activeIndex = null;

      function renderTokens() {{
        strip.innerHTML = "";
        data.tokens.forEach((tok, idx) => {{
          const span = document.createElement("span");
          span.className = "token";
          span.dataset.index = idx;
          span.textContent = tok || "∅";
          span.addEventListener("mouseenter", () => highlightPoint(idx));
          span.addEventListener("mouseleave", () => clearHighlight());
          span.addEventListener("click", () => lockHighlight(idx));
          strip.appendChild(span);
        }});
      }}

      const ctx = document.getElementById("metricChart").getContext("2d");
      const chart = new Chart(ctx, {{
        type: "line",
        data: {{
          labels: data.indices,
          datasets: [
            {{
              label: "Entropy (nats)",
              data: data.entropies,
              borderColor: "#ff7f0e",
              backgroundColor: "rgba(255,127,14,0.15)",
              tension: 0.25,
              pointRadius: 3,
            }},
            {{
              label: "Logprob (ln)",
              data: data.logprobs,
              borderColor: "#1f77b4",
              backgroundColor: "rgba(31,119,180,0.15)",
              tension: 0.2,
              pointRadius: 3,
            }}
          ]
        }},
        options: {{
          interaction: {{ mode: "nearest", intersect: false }},
          responsive: true,
          plugins: {{
            legend: {{ position: "top" }},
            tooltip: {{
              callbacks: {{
                title: (items) => "Token #" + items[0].dataIndex,
                label: (item) => {{
                  const idx = item.dataIndex;
                  return [
                    "Token: " + (data.tokens[idx] || "∅"),
                    "Entropy: " + data.entropies[idx].toFixed(4),
                    "Logprob: " + data.logprobs[idx].toFixed(4),
                  ];
                }}
              }}
            }}
          }},
          scales: {{
            x: {{ title: {{ display: true, text: "Token index" }} }},
            y: {{ title: {{ display: true, text: "Value" }} }}
          }}
        }}
      }});

      function updateMetrics(idx) {{
        metrics.textContent = "Token #" + idx +
          " | token='" + (data.tokens[idx] || "∅") +
          "' | entropy=" + data.entropies[idx].toFixed(4) +
          " | logprob=" + data.logprobs[idx].toFixed(4);
      }}

      function highlightPoint(idx) {{
        if (activeIndex !== null) {{
          strip.children[activeIndex].classList.remove("active");
        }}
        strip.children[idx].classList.add("active");
        activeIndex = idx;
        updateMetrics(idx);
      }}

      function clearHighlight() {{
        if (activeIndex !== null) {{
          strip.children[activeIndex].classList.remove("active");
          activeIndex = null;
        }}
        metrics.textContent = "Hover a token below to view logprob & entropy.";
      }}

      function lockHighlight(idx) {{
        highlightPoint(idx);
      }}

      renderTokens();
    </script>
  </body>
</html>
"""
    output_path.write_text(html_doc, encoding="utf-8")


def main() -> None:
    args = parse_args()
    result = run_generation(args)

    if args.plot_path:
        save_metric_plot(
            result["token_logprobs"],
            Path(args.plot_path),
            args.plot_limit if args.plot_limit and args.plot_limit > 0 else None,
            metric=args.plot_metric,
        )
    if args.html_path:
        save_html_report(
            result["token_logprobs"],
            result["response_text"],
            Path(args.html_path),
        )

    output = json.dumps(result, ensure_ascii=False, indent=2)
    print(output)
    if args.output_json:
        Path(args.output_json).write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
