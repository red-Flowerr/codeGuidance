from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol


@dataclass(slots=True)
class LLMToken:
    """Single token emitted by the LLM with log probability information."""

    text: str
    logprob: float
    top_logprobs: Optional[Dict[str, float]] = None


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


def _tokens_from_logprobs(logprobs_payload: Dict[str, Any]) -> list[LLMToken]:
    tokens: list[LLMToken] = []
    for token_info in logprobs_payload.get("tokens", []):
        token_text = token_info.get("token", "")
        logprob = float(token_info.get("logprob", 0.0))
        top_logprobs_payload = token_info.get("top_logprobs")
        top_logprobs = None
        if isinstance(top_logprobs_payload, list):
            top_logprobs = {
                entry.get("token", ""): float(entry.get("logprob", 0.0))
                for entry in top_logprobs_payload
            }
        tokens.append(LLMToken(text=token_text, logprob=logprob, top_logprobs=top_logprobs))
    return tokens
