"""Typed data structures used by the structured memory system."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional


def _default_timestamp() -> str:
    return datetime.utcnow().isoformat()


@dataclass
class TopicCandidate:
    """A candidate topic/subtopic detected in a conversation window."""

    topic: str
    subtopic: str
    score: float | None = None
    summary: Optional[str] = None

    def to_payload(self) -> Mapping[str, Any]:
        payload: Dict[str, Any] = {"topic": self.topic, "subtopic": self.subtopic}
        if self.score is not None:
            payload["score"] = self.score
        if self.summary:
            payload["summary"] = self.summary
        return payload


@dataclass
class TopicNode:
    """Represents a topic/subtopic stored in the persistent topic database."""

    id: str
    topic: str
    subtopic: str
    embedding: Optional[List[float]] = None
    event_summary: Optional[str] = None
    event_summary_embedding: Optional[List[float]] = None
    metadata: MutableMapping[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Mapping[str, Any]:
        data = asdict(self)
        if self.embedding is not None:
            data["embedding"] = self.embedding
        return data


@dataclass
class ConversationEvent:
    """A single message (event) in the temporary conversation window."""

    conv_id: str
    index: int
    role: str
    content: str
    timestamp: str = field(default_factory=_default_timestamp)
    metadata: MutableMapping[str, Any] = field(default_factory=dict)
    topic_candidates: List[TopicCandidate] = field(default_factory=list)

    def to_payload(self) -> Mapping[str, Any]:
        return {
            "conv_id": self.conv_id,
            "index": self.index,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "topic_candidates": [candidate.to_payload() for candidate in self.topic_candidates],
        }


@dataclass
class ConversationWindow:
    """Sliding window that aggregates the latest conversation events."""

    window_id: str
    events: List[ConversationEvent] = field(default_factory=list)
    summary: Optional[str] = None

    def append(self, event: ConversationEvent) -> None:
        self.events.append(event)

    def __iter__(self) -> Iterable[ConversationEvent]:
        return iter(self.events)

    def to_payload(self) -> Mapping[str, Any]:
        return {
            "window_id": self.window_id,
            "events": [event.to_payload() for event in self.events],
            "summary": self.summary,
        }


@dataclass
class RecallRequest:
    """A request asking the agent to recall an existing topic/subtopic."""

    topic: str
    subtopic: str
    reason: str | None = None

    def to_payload(self) -> Mapping[str, Any]:
        payload: Dict[str, Any] = {"topic": self.topic, "subtopic": self.subtopic}
        if self.reason:
            payload["reason"] = self.reason
        return payload


@dataclass
class MemoryUpdate:
    """Payload describing how a memory item should be updated."""

    topic: str
    subtopic: str
    conv_id: str
    event_index: int
    content: str
    action: str = "add"
    recall: bool = False
    metadata: MutableMapping[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Mapping[str, Any]:
        payload = asdict(self)
        payload["metadata"] = dict(self.metadata)
        return payload


def dumps_payload(data: Mapping[str, Any]) -> str:
    """Render ``data`` as formatted JSON for prompt injection."""

    return json.dumps(data, ensure_ascii=False, indent=2)


__all__ = [
    "ConversationEvent",
    "ConversationWindow",
    "MemoryUpdate",
    "RecallRequest",
    "TopicCandidate",
    "TopicNode",
    "dumps_payload",
]
