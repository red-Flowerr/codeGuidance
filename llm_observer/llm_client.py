from __future__ import annotations

import time
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore


@dataclass(slots=True)
class LLMToken:
    """Single token emitted by the LLM with log probability information."""

    text: str
    logprob: float
    top_logprobs: Optional[Dict[str, float]] = None
    is_special: bool = False


@dataclass(slots=True)
class LLMGeneration:
    """Container for the generated output."""

    text: str
    tokens: list[LLMToken]
    latency_ms: float
    raw_response: Any


class LLMClient(Protocol):
    """Protocol for LLM clients."""

    def generate(self, prompt: str, **kwargs: Any) -> LLMGeneration:
        ...


class OpenAIClient:
    """Wrapper around the OpenAI API that exposes logprobs for each token."""

    def __init__(
        self,
        model: str,
        *,
        client_kwargs: Optional[Dict[str, Any]] = None,
        request_kwargs: Optional[Dict[str, Any]] = None,
        use_responses_api: bool = True,
    ) -> None:
        self.model = model
        self._client_kwargs = client_kwargs or {}
        self._request_kwargs = request_kwargs or {}
        self._client = None
        self._use_responses_api = use_responses_api

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "The 'openai' package is required to use OpenAIClient."
                ) from exc
            self._client = OpenAI(**self._client_kwargs)
        return self._client

    def generate(self, prompt: str, **kwargs: Any) -> LLMGeneration:
        client = self._ensure_client()
        request_kwargs = {**self._request_kwargs, **kwargs}

        start_time = time.perf_counter()
        if self._use_responses_api:
            response = client.responses.create(
                model=self.model,
                input=prompt,
                logprobs=True,
                **request_kwargs,
            )
        else:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                logprobs=True,
                **request_kwargs,
            )
        latency_ms = (time.perf_counter() - start_time) * 1000

        response_dict = (
            response.model_dump() if hasattr(response, "model_dump") else response
        )
        text, tokens = self._extract_output(response_dict)
        return LLMGeneration(text=text, tokens=tokens, latency_ms=latency_ms, raw_response=response_dict)

    @staticmethod
    def _extract_output(response: Dict[str, Any]) -> tuple[str, list[LLMToken]]:
        """Extract text and tokens from Responses or Chat API payloads."""
        if "output" in response:
            return OpenAIClient._extract_from_responses_api(response)
        if "choices" in response:
            return OpenAIClient._extract_from_chat_api(response)
        raise ValueError(
            "Unable to extract content from OpenAI response. Unexpected payload structure."
        )

    @staticmethod
    def _extract_from_responses_api(response: Dict[str, Any]) -> tuple[str, list[LLMToken]]:
        text_parts: list[str] = []
        tokens: list[LLMToken] = []
        for output in response.get("output", []):
            for content in output.get("content", []):
                if content.get("type") == "output_text":
                    text_parts.append(content.get("text", ""))
                    logprobs = content.get("logprobs", {})
                    tokens.extend(_tokens_from_logprobs(logprobs))
        return ("".join(text_parts), tokens)

    @staticmethod
    def _extract_from_chat_api(response: Dict[str, Any]) -> tuple[str, list[LLMToken]]:
        choices = response.get("choices", [])
        if not choices:
            return ("", [])
        choice = choices[0]
        message = choice.get("message", {})
        text = message.get("content", "") or ""
        logprobs = choice.get("logprobs", {})
        tokens = []
        for item in logprobs.get("content", []):
            tokens.extend(_tokens_from_logprobs(item))
        return (text, tokens)


class EchoClient:
    """Utility client for testing; echoes the prompt back."""

    def generate(self, prompt: str, **_: Any) -> LLMGeneration:
        token = LLMToken(text=prompt, logprob=0.0)
        return LLMGeneration(
            text=prompt,
            tokens=[token],
            latency_ms=0.0,
            raw_response={"echo": prompt},
        )


class LocalHFClient:
    """Local HuggingFace transformers client that exposes token log probabilities."""

    def __init__(
        self,
        model_path: str,
        *,
        tokenizer_path: Optional[str] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
    ) -> None:
        if AutoModelForCausalLM is None or AutoTokenizer is None or torch is None:
            raise RuntimeError(
                "The 'transformers' and 'torch' packages are required for LocalHFClient."
            )

        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.generation_kwargs = generation_kwargs or {}
        self.top_k = top_k

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self._maybe_set_padding_token()

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device_map = "auto" if torch.cuda.is_available() else None
        load_kwargs: Dict[str, Any] = {"torch_dtype": dtype}
        if device_map:
            load_kwargs["device_map"] = device_map
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, **load_kwargs
        )
        if device_map is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
        self.model.eval()

    def _maybe_set_padding_token(self) -> None:
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    def generate(self, prompt: str, **kwargs: Any) -> LLMGeneration:
        if torch is None:
            raise RuntimeError("PyTorch is required to use LocalHFClient.")

        request_kwargs = {**self.generation_kwargs, **kwargs}
        max_output_tokens = request_kwargs.pop("max_output_tokens", None)
        if max_output_tokens is None:
            max_output_tokens = request_kwargs.pop("max_new_tokens", None)
        if max_output_tokens is None:
            max_output_tokens = 256

        temperature = request_kwargs.pop("temperature", None)
        do_sample = temperature is not None and temperature > 0.0
        if temperature is None or temperature <= 0.0:
            temperature = None

        tokenizer = self.tokenizer
        model = self.model

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)

        generation_params: Dict[str, Any] = {
            "max_new_tokens": max_output_tokens,
            "return_dict_in_generate": True,
            "output_scores": True,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if do_sample:
            generation_params["do_sample"] = True
            if temperature is not None:
                generation_params["temperature"] = temperature
        generation_params.update(request_kwargs)

        start_time = time.perf_counter()
        with torch.no_grad():
            output = model.generate(input_ids, attention_mask=attention_mask, **generation_params)
        latency_ms = (time.perf_counter() - start_time) * 1000

        sequences = output.sequences[0]
        generated_ids = sequences[input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        tokens: list[LLMToken] = []
        scores = output.scores or []
        special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
        for step, token_id in enumerate(generated_ids):
            if step >= len(scores):
                break
            logits = scores[step][0]
            logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
            token_index = int(token_id.item()) if hasattr(token_id, "item") else int(token_id)
            token_logprob = float(logprobs[token_index])
            top_logprobs_values, top_logprobs_indices = torch.topk(
                logprobs, k=min(self.top_k, logprobs.shape[-1])
            )
            top_logprobs: Dict[str, float] = {}
            for idx, value in zip(top_logprobs_indices.cpu().tolist(), top_logprobs_values.cpu().tolist()):
                prob_val = float(value)
                if math.isinf(prob_val) and prob_val < 0:
                    continue
                decoded = tokenizer.decode([idx], skip_special_tokens=False)
                top_logprobs[decoded] = prob_val
            token_text = tokenizer.decode([token_index], skip_special_tokens=False)
            tokens.append(
                LLMToken(
                    text=token_text,
                    logprob=token_logprob,
                    top_logprobs=top_logprobs,
                    is_special=token_index in special_ids,
                )
            )

        raw_response = {
            "prompt": prompt,
            "generated_ids": generated_ids.tolist(),
            "generated_text": generated_text,
        }
        return LLMGeneration(
            text=generated_text,
            tokens=tokens,
            latency_ms=latency_ms,
            raw_response=raw_response,
        )


def _tokens_from_logprobs(logprobs_payload: Dict[str, Any]) -> list[LLMToken]:
    tokens: list[LLMToken] = []
    for token_info in logprobs_payload.get("tokens", []):
        token_text = token_info.get("token", "")
        logprob = float(token_info.get("logprob", 0.0))
        top_logprobs_payload = token_info.get("top_logprobs")
        top_logprobs = None
        if isinstance(top_logprobs_payload, list):
            top_logprobs = {}
            for entry in top_logprobs_payload:
                logprob_val = float(entry.get("logprob", 0.0))
                if math.isinf(logprob_val) and logprob_val < 0:
                    continue
                top_logprobs[entry.get("token", "")] = logprob_val
        tokens.append(
            LLMToken(
                text=token_text,
                logprob=logprob,
                top_logprobs=top_logprobs,
                is_special=False,
            )
        )
    return tokens
