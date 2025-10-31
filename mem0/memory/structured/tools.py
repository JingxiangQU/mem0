from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from .schemas import ConversationWindow, MemoryUpdate, TopicCandidate
from .storage import TopicDatabase

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    name: str
    payload: MutableMapping[str, object]


@dataclass
class TopicRetrievalTool:
    """Use embeddings to retrieve candidate topics for a conversation window."""

    db: TopicDatabase
    embed_texts: Callable[[Iterable[str]], List[List[float]]]
    top_k: int = 5

    def __call__(self, window: ConversationWindow) -> MutableMapping[str, List[Mapping[str, object]]]:
        if not window.events:
            return {}

        texts = [event.content for event in window.events]
        embeddings = self.embed_texts(texts)
        if len(embeddings) != len(texts):
            logger.warning("Embedding count %s mismatches text count %s", len(embeddings), len(texts))
            embeddings = embeddings[: len(texts)]

        topic_map: Dict[str, Dict[str, Dict[str, object]]] = {}
        for event, embedding in zip(window.events, embeddings):
            results = self.db.search_topics(embedding=embedding, top_k=self.top_k)
            candidates: List[TopicCandidate] = []
            for result in results:
                summary = result.topic.event_summary
                candidates.append(
                    TopicCandidate(
                        topic=result.topic.topic,
                        subtopic=result.topic.subtopic,
                        score=result.score,
                        summary=summary,
                    )
                )
                topic_entry = topic_map.setdefault(result.topic.topic, {})
                payload = {
                    "topic": result.topic.topic,
                    "subtopic": result.topic.subtopic,
                    "summary": summary or "",
                    "score": result.score,
                }
                existing = topic_entry.get(result.topic.subtopic)
                if not existing or payload["score"] > existing.get("score", float("-inf")):
                    topic_entry[result.topic.subtopic] = payload
            event.topic_candidates = candidates

        exported: MutableMapping[str, List[Mapping[str, object]]] = {}
        for topic, subtopics in topic_map.items():
            exported[topic] = sorted(
                subtopics.values(),
                key=lambda item: item.get("score", 0.0),
                reverse=True,
            )
        return exported


@dataclass
class TopicUpdateTool:
    """Persist memory updates and refresh topic summaries."""

    db: TopicDatabase
    embed_texts: Callable[[Iterable[str]], List[List[float]]]
    summarize_event: Callable[[Optional[str], Sequence[str]], Optional[str]]

    def __call__(self, updates: Iterable[MemoryUpdate]) -> List[str]:
        updates = list(updates)
        if not updates:
            return []

        grouped: Dict[Tuple[str, str], List[MemoryUpdate]] = {}
        for update in updates:
            grouped.setdefault((update.topic, update.subtopic), []).append(update)

        topic_texts = [f"{topic}::{subtopic}" for topic, subtopic in grouped]
        topic_embeddings = self.embed_texts(topic_texts) if topic_texts else []
        content_texts = [update.content for update in updates]
        content_embeddings = self.embed_texts(content_texts) if content_texts else []

        topic_ids: Dict[Tuple[str, str], str] = {}
        for idx, ((topic_name, subtopic_name), group_updates) in enumerate(grouped.items()):
            topic_embedding = topic_embeddings[idx] if idx < len(topic_embeddings) else None
            metadata: Dict[str, object] = {}
            for update in group_updates:
                metadata.update(dict(update.metadata))
            topic_id = self.db.upsert_topic(
                topic=topic_name,
                subtopic=subtopic_name,
                embedding=topic_embedding,
                metadata=metadata or None,
            )
            topic_ids[(topic_name, subtopic_name)] = topic_id

        memory_ids: List[str] = []
        for index, update in enumerate(updates):
            topic_id = topic_ids[(update.topic, update.subtopic)]
            content_embedding = content_embeddings[index] if index < len(content_embeddings) else None
            payload_metadata = {"action": update.action, **dict(update.metadata)}
            memory_id = self.db.add_memory(
                topic_id=topic_id,
                conv_id=update.conv_id,
                event_index=update.event_index,
                content=update.content,
                embedding=content_embedding,
                metadata=payload_metadata,
            )
            memory_ids.append(memory_id)

        return memory_ids

    def update_topic_summary(
        self, topic: str, subtopic: str, new_event_texts: Sequence[str]
    ) -> Optional[str]:
        topic_node = self.db.fetch_topic(topic, subtopic)
        previous_summary = topic_node.event_summary if topic_node else None
        summary = self.summarize_event(previous_summary, new_event_texts)
        if summary is None:
            return previous_summary
        summary_embedding = self.embed_texts([summary])[0] if summary else None
        self.db.upsert_topic(
            topic=topic,
            subtopic=subtopic,
            event_summary=summary,
            event_summary_embedding=summary_embedding,
        )
        return summary


@dataclass
class SummaryTool:
    """Create a simple summary of the temporary window."""

    max_sentences: int = 3

    def __call__(self, window: ConversationWindow) -> str:
        texts = [event.content.strip() for event in window.events if event.content.strip()]
        summary = " ".join(texts)
        sentences = summary.split("。") if "。" in summary else summary.split(".")
        filtered = [sentence.strip() for sentence in sentences if sentence.strip()]
        return "。".join(filtered[: self.max_sentences]) if filtered else summary


__all__ = [
    "SummaryTool",
    "ToolResult",
    "TopicRetrievalTool",
    "TopicUpdateTool",
]
