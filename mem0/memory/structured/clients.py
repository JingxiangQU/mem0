"""Unified OpenAI-compatible clients for chat completions and embeddings."""

from __future__ import annotations

import logging
import os
from copy import deepcopy
from typing import Any, Iterable, List, Mapping, MutableMapping, Sequence

from openai import OpenAI

logger = logging.getLogger(__name__)


DEFAULT_EXTRA_BODY: Mapping[str, Any] = {
    "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}
}


class LLMClient:
    """Thin wrapper over :class:`openai.OpenAI` with provider defaults."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        provider: str = "vllm",
        api_key: str | None = None,
        api_key_env: str | None = None,
        default_extra_body: Mapping[str, Any] | None = None,
    ) -> None:
        provider_key = provider.lower()
        if provider_key not in {"vllm", "deepseek", "openai"}:
            raise ValueError(f"Unsupported provider '{provider}'")

        if api_key is None:
            env_name = api_key_env or (
                "DEEPSEEK_API_KEY" if provider_key == "deepseek" else "OPENAI_API_KEY"
            )
            api_key = os.environ.get(env_name) or ""

        extra = default_extra_body
        if extra is None and provider_key == "vllm":
            extra = DEFAULT_EXTRA_BODY

        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.provider = provider_key
        self.default_extra_body = dict(extra or {})

    # ------------------------------------------------------------------
    # Chat completions
    # ------------------------------------------------------------------
    def chat(
        self,
        messages: Sequence[Mapping[str, object]],
        *,
        extra_body: Mapping[str, Any] | None = None,
    ) -> str:
        payload: MutableMapping[str, Any] = {"model": self.model, "messages": list(messages)}
        merged = self._merge_extra(extra_body)
        if merged:
            payload.update(merged)

        logger.debug("Dispatching chat request: %s", payload)
        response = self._client.chat.completions.create(**payload)
        logger.debug("Chat raw response: %s", response)
        choice = response.choices[0].message
        return getattr(choice, "content", "") or ""

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        items = list(texts)
        if not items:
            return []

        payload: MutableMapping[str, Any] = {"model": self.model, "input": items}
        logger.debug("Dispatching embedding request: %s", payload)
        response = self._client.embeddings.create(**payload)
        logger.debug("Embedding raw response: %s", response)
        vectors: List[List[float]] = []
        for entry in response.data:
            vector = getattr(entry, "embedding", None)
            if vector is None:
                continue
            vectors.append([float(x) for x in vector])
        if len(vectors) != len(items):
            logger.warning(
                "Embedding count mismatch: expected %s, received %s", len(items), len(vectors)
            )
        return vectors

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _merge_extra(
        self, extra_body: Mapping[str, Any] | None
    ) -> MutableMapping[str, Any] | None:
        if not self.default_extra_body and not extra_body:
            return None
        merged: MutableMapping[str, Any] = deepcopy(self.default_extra_body)
        if extra_body:
            for key, value in extra_body.items():
                if (
                    key in merged
                    and isinstance(merged[key], MutableMapping)
                    and isinstance(value, Mapping)
                ):
                    merged[key].update(value)  # type: ignore[arg-type]
                else:
                    merged[key] = deepcopy(value) if isinstance(value, Mapping) else value
        return merged


__all__ = ["LLMClient"]
