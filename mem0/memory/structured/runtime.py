"""Runtime helpers for deploying the structured memory system with local models.

This module wires the structured memory manager to locally hosted vLLM servers
for both chat completions and embeddings.  It exposes:

* :class:`StructuredMemoryRuntime` – a convenience wrapper that keeps a
  long-lived :class:`~mem0.memory.structured.manager.StructuredMemoryAgent`
  connected to SQLite storage.
* :func:`main` – a CLI entry point that reads conversation events from JSON
  lines (``{"role": ..., "content": ...}``) and commits memories to disk.

The defaults align with the commands provided by the user:

* LLM: ``python -m vllm.entrypoints.openai.api_server --model /mnt/data/models/Qwen3-8B``
  exposed on ``http://localhost:1109``
* Embedding: ``python -m vllm.entrypoints.openai.api_server --model /mnt/data/models/Qwen3-Embedding-8B``
  exposed on ``http://localhost:1108``

Both payloads include ``{"extra_body": {"chat_template_kwargs": {"enable_thinking":
False}}}`` so that Qwen's reasoning mode is disabled.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Optional

from .clients import (
    ExternalLLMClient,
    ExternalLLMConfig,
    LocalEmbeddingClient,
    LocalEmbeddingConfig,
    LocalLLMClient,
    LocalLLMConfig,
)
from .manager import StructuredMemoryAgent, StructuredMemoryManager
from .storage import TopicDatabase

logger = logging.getLogger(__name__)


@dataclass
class StructuredMemoryRuntime:
    """High level runtime for interacting with the structured memory manager."""

    db_path: str = "structured_memory.sqlite"
    conv_id: str = "session-1"
    llm_base_url: str = "http://localhost:1109"
    llm_path: str = "/v1/chat/completions"
    llm_model: str = "Qwen3-8B"
    llm_provider: str = "local"
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    embedding_base_url: str = "http://localhost:1108"
    embedding_path: str = "/v1/embeddings"
    embedding_model: str = "Qwen3-Embedding-8B"
    window_size: int = 6
    context: MutableMapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        db_parent = Path(self.db_path).expanduser().resolve().parent
        db_parent.mkdir(parents=True, exist_ok=True)

        llm_client: LocalLLMClient | ExternalLLMClient
        if self.llm_provider.lower() == "deepseek":
            llm_client = ExternalLLMClient(
                config=ExternalLLMConfig(
                    base_url=self.deepseek_base_url,
                    model=self.deepseek_model,
                )
            )
        else:
            llm_config = LocalLLMConfig(
                base_url=self.llm_base_url,
                generate_path=self.llm_path,
                model=self.llm_model,
            )
            llm_client = LocalLLMClient(config=llm_config)
        embedding_config = LocalEmbeddingConfig(
            base_url=self.embedding_base_url,
            embed_path=self.embedding_path,
            model=self.embedding_model,
        )

        self.database = TopicDatabase(str(Path(self.db_path).expanduser()))
        self.manager = StructuredMemoryManager(
            db=self.database,
            llm_client=llm_client,
            embedding_client=LocalEmbeddingClient(config=embedding_config),
            window_size=self.window_size,
            context=self.context,
        )
        self.agent = StructuredMemoryAgent(manager=self.manager)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def ingest_event(
        self,
        *,
        role: str,
        content: str,
        conv_id: Optional[str] = None,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> None:
        """Feed a single conversation event into the runtime."""

        self.agent.ingest(
            conv_id=conv_id or self.conv_id,
            role=role,
            content=content,
            metadata=metadata,
        )

    def flush(self) -> Mapping[str, object]:
        """Process the current window and persist any resulting memories."""

        return self.agent.run()


def _iter_events(stream: Iterable[str]) -> Iterable[Mapping[str, object]]:
    for raw_line in stream:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:  # pragma: no cover - CLI guard
            logger.error("Skipping malformed JSON line: %s", line)
            raise SystemExit(1) from exc
        if not isinstance(event, Mapping) or "role" not in event or "content" not in event:
            logger.error("Each line must include 'role' and 'content' fields: %s", line)
            raise SystemExit(1)
        yield event


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the structured memory runtime")
    parser.add_argument("--db", default="structured_memory.sqlite", help="SQLite file for storing memories")
    parser.add_argument("--conv-id", default="session-1", help="Conversation/session identifier")
    parser.add_argument("--llm-url", default="http://localhost:1109", help="Base URL of the local LLM server")
    parser.add_argument(
        "--llm-path",
        default="/v1/chat/completions",
        help="Relative path for the chat completions endpoint",
    )
    parser.add_argument("--llm-model", default="Qwen3-8B", help="LLM model name exposed by the server")
    parser.add_argument(
        "--llm-provider",
        choices=["local", "deepseek"],
        default="local",
        help="选择使用本地模型还是调用 DeepSeek OpenAI 接口",
    )
    parser.add_argument(
        "--deepseek-base-url",
        default="https://api.deepseek.com",
        help="DeepSeek OpenAI 接口地址",
    )
    parser.add_argument(
        "--deepseek-model",
        default="deepseek-chat",
        help="DeepSeek 模型名称",
    )
    parser.add_argument("--embed-url", default="http://localhost:1108", help="Base URL of the embedding server")
    parser.add_argument(
        "--embed-path",
        default="/v1/embeddings",
        help="Relative path for the embedding endpoint",
    )
    parser.add_argument(
        "--embed-model",
        default="Qwen3-Embedding-8B",
        help="Embedding model name exposed by the server",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=6,
        help="Number of recent events to keep in the processing window",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Optional path to a JSONL file. Defaults to reading from standard input.",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=1,
        help="Flush and persist memories after this many ingested events. Set to 0 to only flush at the end.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging to trace prompt/response payloads.",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    runtime = StructuredMemoryRuntime(
        db_path=str(args.db),
        conv_id=args.conv_id,
        llm_base_url=args.llm_url,
        llm_path=args.llm_path,
        llm_model=args.llm_model,
        llm_provider=args.llm_provider,
        deepseek_base_url=args.deepseek_base_url,
        deepseek_model=args.deepseek_model,
        embedding_base_url=args.embed_url,
        embedding_path=args.embed_path,
        embedding_model=args.embed_model,
        window_size=args.window_size,
    )

    def _run_stream(stream: Iterable[str]) -> list[Mapping[str, object]]:
        pending = 0
        collected: list[Mapping[str, object]] = []
        for event in _iter_events(stream):
            runtime.ingest_event(
                role=str(event["role"]),
                content=str(event["content"]),
                conv_id=event.get("conv_id"),
                metadata=event.get("metadata"),
            )
            pending += 1
            if args.flush_every and pending >= args.flush_every:
                collected.append(runtime.flush())
                pending = 0
        if pending or not args.flush_every:
            collected.append(runtime.flush())
        return collected

    if args.input:
        with args.input.open("r", encoding="utf-8") as fh:
            results = _run_stream(fh)
    else:
        results = _run_stream(sys.stdin)

    for result in results:
        print(json.dumps(result, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
