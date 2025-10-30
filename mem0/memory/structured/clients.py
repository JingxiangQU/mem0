"""Clients for calling locally or remotely deployed language models.

The default workflow targets locally hosted LLM/embedding services (such as
vLLM) that expose OpenAI-compatible HTTP APIs.  In addition to the local
clients, this module provides a thin wrapper around the ``openai`` SDK so that
integrators can also route calls to external providers like DeepSeek without
changing the higher level memory management code.
"""

from __future__ import annotations

import json
import logging
import os
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional
from urllib import request

from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class LocalLLMConfig:
    """Configuration for :class:`LocalLLMClient`."""

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
class ExternalLLMConfig:
    """Configuration for :class:`ExternalLLMClient`."""

    api_key: str | None = field(default_factory=lambda: os.environ.get("DEEPSEEK_API_KEY"))
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    default_extra_body: Mapping[str, Any] | None = field(
        default_factory=lambda: {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    )


@dataclass
class ExternalLLMClient:
    """Client that talks to OpenAI compatible hosted services via ``openai``."""

    config: ExternalLLMConfig = field(default_factory=ExternalLLMConfig)

    def __post_init__(self) -> None:
        if not self.config.api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY environment variable is required for ExternalLLMClient"
            )
        self._client = OpenAI(api_key=self.config.api_key, base_url=self.config.base_url)

    def generate(
        self,
        *,
        prompt: str,
        system_prompt: str | None = None,
        messages: Optional[List[Mapping[str, Any]]] = None,
        extra_payload: Optional[MutableMapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        payload: Dict[str, Any] = {"model": self.config.model}
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

        merged_extra: Dict[str, Any] = {}
        if self.config.default_extra_body:
            merged_extra = deepcopy(self.config.default_extra_body)
        if extra_payload:
            merged_extra = _deep_merge_dicts(merged_extra, extra_payload)
        payload.update(merged_extra)

        logger.debug("Dispatching external LLM request: %s", payload)
        response = self._client.chat.completions.create(**payload)
        logger.debug("External LLM raw response: %s", response)
        return response.model_dump()


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
    "ExternalLLMClient",
    "ExternalLLMConfig",
    "LocalLLMClient",
    "LocalLLMConfig",
]


def _deep_merge_dicts(
    base: MutableMapping[str, Any], incoming: Mapping[str, Any]
) -> MutableMapping[str, Any]:
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
