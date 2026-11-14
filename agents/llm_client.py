"""Provider-agnostic LLM interface. This is the only module allowed to import a provider SDK.

    client = get_client()
    cas = client.extract(system_prompt, document_text, CAS)   # -> validated CAS instance
    client.last_call                                           # usage / latency of that call
"""
from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TypeVar

from pydantic import BaseModel

from config import MODEL

T = TypeVar("T", bound=BaseModel)


class ExtractionError(RuntimeError):
    """The provider returned no usable structured output (refusal, truncation, invalid JSON)."""


@dataclass
class CallInfo:
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    stop_reason: str | None = None
    request_id: str | None = None
    extra: dict = field(default_factory=dict)


class LLMClient(ABC):
    """One method: read a document, return an instance of the given pydantic model."""

    provider: str = "abstract"

    def __init__(self, model: str):
        self.model = model
        self._local = threading.local()  # last_call is per thread, so one client can be shared by workers

    @property
    def last_call(self) -> CallInfo | None:
        """Usage/latency of the most recent extract() made from the calling thread."""
        return getattr(self._local, "last_call", None)

    @last_call.setter
    def last_call(self, info: CallInfo | None) -> None:
        self._local.last_call = info

    @abstractmethod
    def extract(self, system_prompt: str, document_text: str, pydantic_model: type[T]) -> T: ...


class AnthropicClient(LLMClient):
    """Structured outputs: the JSON schema is generated from the pydantic model and enforced by the API."""

    provider = "anthropic"

    def __init__(self, model: str = MODEL, max_tokens: int = 16000):
        super().__init__(model)
        import anthropic  # local import keeps the SDK confined to this module

        self._anthropic = anthropic
        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY (loaded from .env by config)
        self.max_tokens = max_tokens

    def extract(self, system_prompt: str, document_text: str, pydantic_model: type[T]) -> T:
        t0 = time.perf_counter()
        self.last_call = None
        try:
            resp = self._client.messages.parse(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": f"<document>\n{document_text}\n</document>"}],
                output_format=pydantic_model,  # SDK derives the JSON schema and validates the reply
            )
        except self._anthropic.APIStatusError as e:
            raise ExtractionError(f"{self.provider} API error {e.status_code}: {e.message}") from e
        except self._anthropic.APIConnectionError as e:
            raise ExtractionError(f"{self.provider} connection error: {e}") from e
        except ValueError as e:  # pydantic ValidationError subclasses ValueError
            raise ExtractionError(f"output failed {pydantic_model.__name__} validation: {e}") from e

        self.last_call = CallInfo(
            provider=self.provider, model=resp.model,
            input_tokens=resp.usage.input_tokens, output_tokens=resp.usage.output_tokens,
            latency_s=round(time.perf_counter() - t0, 2), stop_reason=resp.stop_reason,
            request_id=getattr(resp, "_request_id", None),
        )
        if resp.stop_reason in ("refusal", "max_tokens") or resp.parsed_output is None:
            raise ExtractionError(f"no structured output (stop_reason={resp.stop_reason})")
        return resp.parsed_output


def get_client(provider: str = "anthropic", model: str = MODEL) -> LLMClient:
    if provider == "anthropic":
        return AnthropicClient(model)
    raise ValueError(f"unknown provider {provider!r}")
