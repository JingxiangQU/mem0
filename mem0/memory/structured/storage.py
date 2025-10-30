"""Persistent storage for the structured topic / memory architecture."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

from .schemas import TopicNode


@dataclass
class TopicSearchResult:
    topic: TopicNode
    score: float


class TopicDatabase:
    """Small SQLite wrapper that stores topics, subtopics, and memory records."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self.connection = sqlite3.connect(db_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._create_schema()

    def _create_schema(self) -> None:
        with self._lock:
            cur = self.connection.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS topics (
                    id TEXT PRIMARY KEY,
                    topic TEXT NOT NULL,
                    subtopic TEXT NOT NULL,
                    embedding BLOB,
                    event_summary TEXT,
                    event_summary_embedding BLOB,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_topic_unique
                ON topics(topic, subtopic)
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    topic_id TEXT NOT NULL,
                    conv_id TEXT NOT NULL,
                    event_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding BLOB,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(topic_id) REFERENCES topics(id)
                )
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memories_topic_id
                ON memories(topic_id)
                """
            )
            self.connection.commit()
            self._ensure_column("topics", "event_summary", "TEXT")
            self._ensure_column("topics", "event_summary_embedding", "BLOB")

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        cur = self.connection.execute(f"PRAGMA table_info({table})")
        columns = {row[1] for row in cur.fetchall()}
        if column not in columns:
            self.connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
            self.connection.commit()

    @staticmethod
    def _serialize_vector(vector: Optional[Sequence[float]]) -> Optional[bytes]:
        if vector is None:
            return None
        return json.dumps([float(x) for x in vector]).encode("utf-8")

    @staticmethod
    def _deserialize_vector(blob: Optional[bytes]) -> Optional[List[float]]:
        if blob is None:
            return None
        return [float(x) for x in json.loads(blob.decode("utf-8"))]

    def upsert_topic(
        self,
        *,
        topic: str,
        subtopic: str,
        embedding: Optional[Sequence[float]] = None,
        event_summary: Optional[str] = None,
        event_summary_embedding: Optional[Sequence[float]] = None,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> str:
        """Insert or update a topic/subtopic pair and return its identifier."""

        now = datetime.utcnow().isoformat()
        with self._lock:
            cur = self.connection.cursor()
            cur.execute(
                """
                SELECT id, metadata, event_summary, event_summary_embedding
                FROM topics WHERE topic = ? AND subtopic = ?
                """,
                (topic, subtopic),
            )
            row = cur.fetchone()
            if row:
                topic_id = row["id"]
                existing_metadata = json.loads(row["metadata"]) if row["metadata"] else {}
                merged_metadata = {**existing_metadata, **(metadata or {})}
                summary_value = event_summary if event_summary is not None else row["event_summary"]
                summary_embedding_blob = (
                    self._serialize_vector(event_summary_embedding)
                    if event_summary_embedding is not None
                    else row["event_summary_embedding"]
                )
                cur.execute(
                    """
                    UPDATE topics
                    SET embedding = ?, metadata = ?, event_summary = ?, event_summary_embedding = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        self._serialize_vector(embedding),
                        json.dumps(merged_metadata, ensure_ascii=False),
                        summary_value,
                        summary_embedding_blob,
                        now,
                        topic_id,
                    ),
                )
            else:
                topic_id = str(uuid.uuid4())
                cur.execute(
                    """
                    INSERT INTO topics(
                        id, topic, subtopic, embedding, event_summary, event_summary_embedding,
                        metadata, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        topic_id,
                        topic,
                        subtopic,
                        self._serialize_vector(embedding),
                        event_summary,
                        self._serialize_vector(event_summary_embedding),
                        json.dumps(metadata or {}, ensure_ascii=False),
                        now,
                        now,
                    ),
                )
            self.connection.commit()
            return topic_id

    def add_memory(
        self,
        *,
        topic_id: str,
        conv_id: str,
        event_index: int,
        content: str,
        embedding: Optional[Sequence[float]] = None,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> str:
        now = datetime.utcnow().isoformat()
        payload_metadata = json.dumps(metadata or {}, ensure_ascii=False)
        with self._lock:
            memory_id = str(uuid.uuid4())
            self.connection.execute(
                """
                INSERT INTO memories(
                    id, topic_id, conv_id, event_index, content,
                    embedding, metadata, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    topic_id,
                    conv_id,
                    event_index,
                    content,
                    self._serialize_vector(embedding),
                    payload_metadata,
                    now,
                    now,
                ),
            )
            self.connection.commit()
            return memory_id

    def get_topic_summary(self, topic_id: str) -> Optional[str]:
        cur = self.connection.execute(
            "SELECT event_summary FROM topics WHERE id = ?",
            (topic_id,),
        )
        row = cur.fetchone()
        return row[0] if row else None

    def update_topic_summary(
        self,
        *,
        topic_id: str,
        summary: Optional[str],
        embedding: Optional[Sequence[float]] = None,
    ) -> None:
        now = datetime.utcnow().isoformat()
        with self._lock:
            self.connection.execute(
                """
                UPDATE topics
                SET event_summary = ?, event_summary_embedding = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    summary,
                    self._serialize_vector(embedding),
                    now,
                    topic_id,
                ),
            )
            self.connection.commit()

    def update_memory(
        self,
        *,
        memory_id: str,
        content: str,
        embedding: Optional[Sequence[float]] = None,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> None:
        now = datetime.utcnow().isoformat()
        payload_metadata = json.dumps(metadata or {}, ensure_ascii=False)
        with self._lock:
            self.connection.execute(
                """
                UPDATE memories
                SET content = ?, embedding = ?, metadata = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    content,
                    self._serialize_vector(embedding),
                    payload_metadata,
                    now,
                    memory_id,
                ),
            )
            self.connection.commit()

    def fetch_topic(self, topic: str, subtopic: str) -> Optional[TopicNode]:
        cur = self.connection.execute(
            "SELECT * FROM topics WHERE topic = ? AND subtopic = ?",
            (topic, subtopic),
        )
        row = cur.fetchone()
        if not row:
            return None
        return TopicNode(
            id=row["id"],
            topic=row["topic"],
            subtopic=row["subtopic"],
            embedding=self._deserialize_vector(row["embedding"]),
            event_summary=row["event_summary"],
            event_summary_embedding=self._deserialize_vector(row["event_summary_embedding"]),
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    def list_topics(self) -> List[TopicNode]:
        cur = self.connection.execute("SELECT * FROM topics")
        rows = cur.fetchall()
        return [
            TopicNode(
                id=row["id"],
                topic=row["topic"],
                subtopic=row["subtopic"],
                embedding=self._deserialize_vector(row["embedding"]),
                event_summary=row["event_summary"],
                event_summary_embedding=self._deserialize_vector(row["event_summary_embedding"]),
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            )
            for row in rows
        ]

    @staticmethod
    def _cosine_similarity(vec1: Sequence[float], vec2: Sequence[float]) -> float:
        if not vec1 or not vec2:
            return 0.0
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    def search_topics(
        self, *, embedding: Sequence[float], top_k: int = 5
    ) -> List[TopicSearchResult]:
        topics = self.list_topics()
        scored: List[TopicSearchResult] = []
        for topic_node in topics:
            if topic_node.embedding is None:
                continue
            score = self._cosine_similarity(embedding, topic_node.embedding)
            scored.append(TopicSearchResult(topic=topic_node, score=score))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def fetch_memories(
        self,
        topic_id: str,
        *,
        query_embedding: Optional[Sequence[float]] = None,
        top_k: Optional[int] = None,
        min_score: float = 0.0,
    ) -> List[Mapping[str, object]]:
        cur = self.connection.execute(
            "SELECT * FROM memories WHERE topic_id = ? ORDER BY created_at ASC",
            (topic_id,),
        )
        rows = cur.fetchall()
        memories: List[Mapping[str, object]] = []
        for row in rows:
            embedding = self._deserialize_vector(row["embedding"])
            similarity = None
            if query_embedding is not None and embedding is not None:
                similarity = self._cosine_similarity(query_embedding, embedding)
            memories.append(
                {
                    "id": row["id"],
                    "topic_id": row["topic_id"],
                    "conv_id": row["conv_id"],
                    "event_index": row["event_index"],
                    "content": row["content"],
                    "embedding": embedding,
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "similarity": similarity,
                }
            )

        if query_embedding is not None:
            filtered = [
                item
                for item in memories
                if item["similarity"] is None or item["similarity"] >= min_score
            ]
            filtered.sort(key=lambda item: item["similarity"] or 0.0, reverse=True)
            if top_k is not None:
                filtered = filtered[:top_k]
            return filtered

        return memories


__all__ = ["TopicDatabase", "TopicSearchResult"]
