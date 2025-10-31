"""Runtime helpers for deploying the structured memory system."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Optional

from .clients import LLMClient
from .manager import StructuredMemoryAgent, StructuredMemoryManager
from .schemas import ConversationEvent, ConversationWindow
from .storage import TopicDatabase

logger = logging.getLogger(__name__)


@dataclass
class StructuredMemoryRuntime:
    """High level runtime for interacting with the structured memory manager."""

    db_path: str = "structured_memory.sqlite"
    llm_url: str = "http://localhost:1109"
    llm_model: str = "Qwen3-8B"
    llm_provider: str = "vllm"
    embed_url: str = "http://localhost:1108"
    embed_model: str = "Qwen3-Embedding-8B"
    embed_provider: str = "vllm"
    window_size: int = 6
    context: MutableMapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        db_parent = Path(self.db_path).expanduser().resolve().parent
        db_parent.mkdir(parents=True, exist_ok=True)

        self.database = TopicDatabase(str(Path(self.db_path).expanduser()))

        self.llm_client = LLMClient(
            base_url=self.llm_url,
            model=self.llm_model,
            provider=self.llm_provider,
        )
        self.embedding_client = LLMClient(
            base_url=self.embed_url,
            model=self.embed_model,
            provider=self.embed_provider,
            default_extra_body={},
        )

        self.manager = StructuredMemoryManager(
            db=self.database,
            llm_client=self.llm_client,
            embedding_client=self.embedding_client,
            context=self.context,
        )
        self.agent = StructuredMemoryAgent(manager=self.manager)

        self._current_window_id: str | None = None
        self._window_events: list[ConversationEvent] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def ingest_event(
        self,
        *,
        role: str,
        content: str,
        conv_id: str,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> None:
        if self._current_window_id is None:
            self._current_window_id = conv_id
        elif conv_id != self._current_window_id:
            self.flush()
            self._current_window_id = conv_id

        event = ConversationEvent(
            conv_id=conv_id,
            index=len(self._window_events),
            role=role,
            content=content,
            metadata=dict(metadata or {}),
        )
        self._window_events.append(event)

    def flush(self) -> Mapping[str, object]:
        if not self._window_events:
            return {"window_id": self._current_window_id, "events": []}

        window = ConversationWindow(
            window_id=self._current_window_id or "window-0",
            events=list(self._window_events),
        )
        result = self.agent.run_tempwindow(window)
        self._window_events.clear()
        self._current_window_id = None
        return result


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
    parser.add_argument("--llm-url", default="http://localhost:1109", help="Base URL of the LLM server")
    parser.add_argument("--llm-model", default="Qwen3-8B", help="LLM model name exposed by the server")
    parser.add_argument(
        "--llm-provider",
        choices=["vllm", "deepseek", "openai"],
        default="vllm",
        help="LLM provider type",
    )
    parser.add_argument("--embed-url", default="http://localhost:1108", help="Base URL of the embedding server")
    parser.add_argument(
        "--embed-model",
        default="Qwen3-Embedding-8B",
        help="Embedding model name exposed by the server",
    )
    parser.add_argument(
        "--embed-provider",
        choices=["vllm", "deepseek", "openai"],
        default="vllm",
        help="Embedding provider type",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Optional path to a JSONL file. Defaults to reading from standard input.",
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
        llm_url=args.llm_url,
        llm_model=args.llm_model,
        llm_provider=args.llm_provider,
        embed_url=args.embed_url,
        embed_model=args.embed_model,
        embed_provider=args.embed_provider,
    )

    def _run_stream(stream: Iterable[str]) -> list[Mapping[str, object]]:
        results: list[Mapping[str, object]] = []
        for event in _iter_events(stream):
            runtime.ingest_event(
                role=str(event["role"]),
                content=str(event["content"]),
                conv_id=str(event.get("conv_id") or "session-1"),
                metadata=event.get("metadata"),
            )
        results.append(runtime.flush())
        return results

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
