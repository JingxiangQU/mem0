"""Clients for calling locally deployed LLM and embedding services.

The design document requires that the memory system communicates with locally
hosted language and embedding models.  The classes below implement thin HTTP
clients that default to ``http://localhost`` endpoints so that integrators can
wire in their preferred runtime (Ollama, vLLM, llama.cpp, etc.).

The implementations rely only on ``urllib`` from the Python standard library so
that no extra dependencies are introduced.  Both clients expose small, well
typed methods that the rest of the memory system can use.
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional
from urllib import request

logger = logging.getLogger(__name__)


@dataclass
class LocalLLMConfig:
    """Configuration for :class:`LocalLLMClient`.

    Attributes:
        base_url: Base URL of the locally deployed LLM service.
        generate_path: Relative path where the generation endpoint is exposed.
        timeout: Network timeout in seconds. Defaults to ``30``.
        default_headers: Optional headers (e.g., auth tokens) added to every
            request.
        model: Optional model identifier passed to the endpoint.
    """

    base_url: str = "http://localhost:1109"
    generate_path: str = "/v1/chat/completions"
    timeout: float = 30.0
    default_headers: Mapping[str, str] | None = None
    model: str | None = None
    default_extra_body: Mapping[str, Any] | None = field(
        default_factory=lambda: {
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}
        }
    )

    def build_url(self) -> str:
        return self.base_url.rstrip("/") + self.generate_path


@dataclass
class LocalLLMClient:
    """Simple JSON based LLM client that targets a locally deployed model."""

    config: LocalLLMConfig = field(default_factory=LocalLLMConfig)

    def generate(
        self,
        *,
        prompt: str,
        system_prompt: str | None = None,
        messages: Optional[List[Mapping[str, Any]]] = None,
        extra_payload: Optional[MutableMapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        """Call the local LLM service with the provided prompt data.

        Args:
            prompt: User prompt (usually the instruction formatted with the
                architecture data structures).
            system_prompt: Optional system prompt.  If provided, the payload uses
                an ``messages`` style schema.  Otherwise ``prompt`` is sent as-is.
            messages: Optional chat style message history.  When supplied it is
                appended after the system prompt and the final user message.
            extra_payload: Extra JSON fields forwarded verbatim.

        Returns:
            Parsed JSON response from the service.
        """

        payload: Dict[str, Any] = {}
        if system_prompt or messages:
            chat_messages: List[Mapping[str, Any]] = []
            if system_prompt:
                chat_messages.append({"role": "system", "content": system_prompt})
            if messages:
                chat_messages.extend(messages)
            chat_messages.append({"role": "user", "content": prompt})
            payload["messages"] = chat_messages
        else:
            payload["prompt"] = prompt

        if self.config.model:
            payload["model"] = self.config.model

        merged_extra: Dict[str, Any] = {}
        if self.config.default_extra_body:
            merged_extra = deepcopy(self.config.default_extra_body)
        if extra_payload:
            merged_extra = _deep_merge_dicts(merged_extra, extra_payload)
        payload.update(merged_extra)

        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.config.build_url(),
            data=body,
            headers={"Content-Type": "application/json", **(self.config.default_headers or {})},
            method="POST",
        )

        logger.debug("Dispatching local LLM request: %s", payload)
        with request.urlopen(req, timeout=self.config.timeout) as resp:
            data = resp.read().decode("utf-8")
        logger.debug("Local LLM raw response: %s", data)
        try:
            return json.loads(data)
        except json.JSONDecodeError as exc:
            logger.error("Failed to decode LLM response: %s", data)
            raise ValueError("Local LLM returned invalid JSON") from exc


@dataclass
class LocalEmbeddingConfig:
    """Configuration for :class:`LocalEmbeddingClient`."""

    base_url: str = "http://localhost:1108"
    embed_path: str = "/v1/embeddings"
    timeout: float = 30.0
    default_headers: Mapping[str, str] | None = None
    model: str | None = None

    def build_url(self) -> str:
        return self.base_url.rstrip("/") + self.embed_path


@dataclass
class LocalEmbeddingClient:
    """Client that retrieves embeddings from a locally deployed encoder."""

    config: LocalEmbeddingConfig = field(default_factory=LocalEmbeddingConfig)

    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        """Embed a batch of texts using the configured local service."""

        texts = list(texts)
        if not texts:
            return []

        payload: Dict[str, Any] = {"input": texts}
        if self.config.model:
            payload["model"] = self.config.model

        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.config.build_url(),
            data=body,
            headers={"Content-Type": "application/json", **(self.config.default_headers or {})},
            method="POST",
        )

        logger.debug("Dispatching local embedding request: %s", payload)
        with request.urlopen(req, timeout=self.config.timeout) as resp:
            data = resp.read().decode("utf-8")
        logger.debug("Local embedding raw response: %s", data)
        try:
            response = json.loads(data)
        except json.JSONDecodeError as exc:
            logger.error("Failed to decode embedding response: %s", data)
            raise ValueError("Embedding endpoint returned invalid JSON") from exc

        vectors: Optional[List[List[float]]] = None
        if isinstance(response.get("embeddings"), list):
            vectors = [
                [float(x) for x in item]
                for item in response["embeddings"]
                if isinstance(item, list)
            ]
        elif isinstance(response.get("vectors"), list):
            vectors = [
                [float(x) for x in item]
                for item in response["vectors"]
                if isinstance(item, list)
            ]
        elif isinstance(response.get("data"), list):
            data_vectors: List[List[float]] = []
            for entry in response["data"]:
                if isinstance(entry, Mapping) and isinstance(entry.get("embedding"), list):
                    data_vectors.append([float(x) for x in entry["embedding"]])
            if data_vectors:
                vectors = data_vectors

        if not vectors:
            raise ValueError("Embedding endpoint response missing embeddings data")

        return vectors


__all__ = [
    "LocalEmbeddingClient",
    "LocalEmbeddingConfig",
    "LocalLLMClient",
    "LocalLLMConfig",
]


def _deep_merge_dicts(
    base: MutableMapping[str, Any], incoming: Mapping[str, Any]
) -> MutableMapping[str, Any]:
    """Recursively merge ``incoming`` dict into ``base``."""

    for key, value in incoming.items():
        if (
            key in base
            and isinstance(base[key], MutableMapping)
            and isinstance(value, Mapping)
        ):
            _deep_merge_dicts(base[key], value)
        else:
            base[key] = deepcopy(value) if isinstance(value, Mapping) else value
    return base
