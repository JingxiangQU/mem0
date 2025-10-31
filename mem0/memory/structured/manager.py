"""High-level orchestration for the structured memory workflow."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

from .clients import LLMClient
from .prompts import (
    SYSTEM_PROMPT_1,
    SYSTEM_PROMPT_2,
    SYSTEM_PROMPT_3,
    SYSTEM_PROMPT_4,
    TOPIC_EVENT0_PROMPT,
)
from .schemas import ConversationEvent, ConversationWindow, MemoryUpdate
from .storage import TopicDatabase
from .tools import SummaryTool, TopicRetrievalTool, TopicUpdateTool

logger = logging.getLogger(__name__)


@dataclass
class StructuredMemoryManager:
    """Manage conversation windows, prompts, and persistence."""

    db: TopicDatabase
    llm_client: LLMClient
    embedding_client: LLMClient
    recall_top_k: int = 5
    context: MutableMapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.summary_tool = SummaryTool()
        self.retrieval_tool = TopicRetrievalTool(
            db=self.db,
            embed_texts=self._safe_embed,
            top_k=self.recall_top_k,
        )
        self.update_tool = TopicUpdateTool(
            db=self.db,
            embed_texts=self._safe_embed,
            summarize_event=self._summarize_event0,
        )

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------
    def _safe_embed(self, texts: Iterable[str]) -> List[List[float]]:
        items = list(texts)
        if not items:
            return []
        try:
            return self.embedding_client.embed(items)
        except Exception as exc:  # pragma: no cover - network failure fallback
            logger.warning("Embedding service unavailable: %s", exc)
            return [[0.0] * 4 for _ in items]

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------
    def _call_prompt1(
        self,
        *,
        event: ConversationEvent,
        context_payload: Sequence[Mapping[str, object]],
        calltopics: Mapping[str, List[Mapping[str, object]]],
        windowtopics: Mapping[str, List[str]],
    ) -> Mapping[str, Any]:
        payload = {
            "conv": {"role": event.role, "content": event.content},
            "context": context_payload,
            "calltopics": calltopics,
            "windowtopics": windowtopics,
        }
        return self._invoke_llm(SYSTEM_PROMPT_1, payload)

    def _call_prompt2(
        self,
        *,
        event: ConversationEvent,
        calltopics: Mapping[str, List[Mapping[str, object]]],
    ) -> Mapping[str, Any]:
        payload = {
            "conv": {"role": event.role, "content": event.content},
            "recalltopics": calltopics,
        }
        return self._invoke_llm(SYSTEM_PROMPT_2, payload)

    def _call_prompt3(
        self,
        *,
        event: ConversationEvent,
        calltopics: Mapping[str, List[Mapping[str, object]]],
    ) -> Mapping[str, Any]:
        payload = {
            "conv": {"role": event.role, "content": event.content},
            "recalltopics": calltopics,
        }
        return self._invoke_llm(SYSTEM_PROMPT_3, payload)

    def _call_prompt4(
        self,
        *,
        event: ConversationEvent,
        topic: str,
        subtopic: str,
        history_summary: Optional[str],
        recalls: Sequence[Mapping[str, object]],
    ) -> Mapping[str, Any]:
        payload = {
            "conv": {"role": event.role, "content": event.content},
            "topic": topic,
            "subtopic": subtopic,
            "history_event0": history_summary,
            "recalls": recalls,
        }
        return self._invoke_llm(SYSTEM_PROMPT_4, payload)

    def _summarize_event0(
        self, history_summary: Optional[str], new_evidence: Sequence[str]
    ) -> Optional[str]:
        evidence = [text for text in new_evidence if text.strip()]
        if not evidence and not (history_summary or ""):
            return None
        payload = {
            "history_summary": history_summary or "",
            "new_evidence": evidence,
        }
        messages = [
            {"role": "system", "content": TOPIC_EVENT0_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        response = self.llm_client.chat(messages)
        return response.strip()

    def _invoke_llm(self, system_prompt: str, payload: Mapping[str, object]) -> Mapping[str, Any]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        raw = self.llm_client.chat(messages)
        parsed = self._extract_json(raw)
        if parsed is None:
            logger.warning("Failed to parse LLM JSON response: %s", raw)
            return {}
        return parsed

    @staticmethod
    def _extract_json(message: str) -> Optional[Mapping[str, Any]]:
        sanitized = message.strip()
        if sanitized.startswith("```"):
            sanitized = sanitized[3:]
            if sanitized.lower().startswith("json"):
                sanitized = sanitized[4:]
            sanitized = sanitized.lstrip("\n")
            if sanitized.endswith("```"):
                sanitized = sanitized[:-3]
        elif sanitized.lower().startswith("json"):
            sanitized = sanitized[4:].lstrip(": ")

        try:
            return json.loads(sanitized)
        except json.JSONDecodeError:
            pass

        start = None
        depth = 0
        in_string = False
        escape = False
        for idx, char in enumerate(sanitized):
            if start is None:
                if char == "{":
                    start = idx
                    depth = 1
            else:
                if in_string:
                    if escape:
                        escape = False
                    elif char == "\\":
                        escape = True
                    elif char == '"':
                        in_string = False
                else:
                    if char == '"':
                        in_string = True
                    elif char == "{":
                        depth += 1
                    elif char == "}":
                        depth -= 1
                        if depth == 0 and start is not None:
                            blob = sanitized[start : idx + 1]
                            return json.loads(blob)
        return None

    # ------------------------------------------------------------------
    # Window orchestration
    # ------------------------------------------------------------------
    def process_window(self, window: ConversationWindow) -> Mapping[str, Any]:
        if not window.events:
            return {"window_id": window.window_id, "events": [], "context": []}

        context_map = self._build_context_for_window(window.window_id)
        calltopics = self._build_calltopics(window)
        windowtopics = self._windowtopics_payload(context_map)
        results: List[Mapping[str, Any]] = []

        for event in window.events:
            context_payload = self._context_payload(context_map)
            decision = self._call_prompt1(
                event=event,
                context_payload=context_payload,
                calltopics=calltopics,
                windowtopics=windowtopics,
            )

            topic, subtopic = self._resolve_topic(event, decision, calltopics)
            history_summary = self._topic_summary(context_map, topic, subtopic)
            recalls = self._recall_memories(topic, subtopic, event.content)
            prompt4 = self._call_prompt4(
                event=event,
                topic=topic,
                subtopic=subtopic,
                history_summary=history_summary,
                recalls=recalls,
            )
            summary_text = (prompt4.get("summary") or "").strip()

            metadata = {
                "evidence_turns": [event.index],
                "source_role": event.role,
                "raw_content": event.content,
            }
            update_payload = {
                "topic": topic,
                "subtopic": subtopic,
                "conv_id": event.conv_id,
                "event_index": event.index,
                "summary": summary_text,
                "metadata": metadata,
            }
            sanitized = self._sanitize_memory_update_payload(update_payload)
            memory_update = MemoryUpdate(**sanitized)
            stored_ids = self.update_tool([memory_update])

            updated_summary = self.update_tool.update_topic_summary(
                topic, subtopic, [summary_text] if summary_text else []
            )
            self._refresh_context(context_map, topic, subtopic, updated_summary or history_summary, event.index)
            windowtopics = self._windowtopics_payload(context_map)

            results.append(
                {
                    "event_index": event.index,
                    "topic": topic,
                    "subtopic": subtopic,
                    "summary": summary_text,
                    "stored_memory_ids": stored_ids,
                }
            )

        return {
            "window_id": window.window_id,
            "events": results,
            "context": self._context_payload(context_map),
        }

    # ------------------------------------------------------------------
    # Context helpers
    # ------------------------------------------------------------------
    def _build_context_for_window(
        self, conv_id: str
    ) -> MutableMapping[str, MutableMapping[str, MutableMapping[str, Any]]]:
        context: MutableMapping[str, MutableMapping[str, MutableMapping[str, Any]]] = {}
        for node in self.db.list_topics():
            if not node.event_summary:
                continue
            topic_entry = context.setdefault(node.topic, {})
            topic_entry[node.subtopic] = {
                "summary": node.event_summary,
                "turns": [],
            }
        return context

    def _build_calltopics(
        self, window: ConversationWindow
    ) -> MutableMapping[str, List[Mapping[str, object]]]:
        return self.retrieval_tool(window)

    def _windowtopics_payload(
        self, context_map: MutableMapping[str, MutableMapping[str, MutableMapping[str, Any]]]
    ) -> MutableMapping[str, List[str]]:
        payload: MutableMapping[str, List[str]] = {}
        for topic, subtopics in context_map.items():
            selections = [name for name, info in subtopics.items() if info.get("turns")]
            if selections:
                payload[topic] = sorted(selections)
        return payload

    def _context_payload(
        self, context_map: MutableMapping[str, MutableMapping[str, MutableMapping[str, Any]]]
    ) -> List[Mapping[str, object]]:
        exported: List[Mapping[str, object]] = []
        for topic, subtopics in context_map.items():
            for name, info in subtopics.items():
                exported.append(
                    {
                        "topic": topic,
                        "subtopic": name,
                        "summary": info.get("summary"),
                        "turns": info.get("turns", []),
                    }
                )
        return exported

    def _refresh_context(
        self,
        context_map: MutableMapping[str, MutableMapping[str, MutableMapping[str, Any]]],
        topic: str,
        subtopic: str,
        summary: Optional[str],
        event_index: int,
    ) -> None:
        topic_entry = context_map.setdefault(topic, {})
        entry = topic_entry.setdefault(subtopic, {"summary": None, "turns": []})
        if summary:
            entry["summary"] = summary
        turns = entry.setdefault("turns", [])
        if event_index not in turns:
            turns.append(event_index)

    def _topic_summary(
        self,
        context_map: MutableMapping[str, MutableMapping[str, MutableMapping[str, Any]]],
        topic: str,
        subtopic: str,
    ) -> Optional[str]:
        return context_map.get(topic, {}).get(subtopic, {}).get("summary")

    def _recall_memories(
        self, topic: str, subtopic: str, content: str
    ) -> List[Mapping[str, object]]:
        node = self.db.fetch_topic(topic, subtopic)
        if not node:
            return []
        embedding = None
        try:
            embedding = self.embedding_client.embed([content])[0]
        except Exception:  # pragma: no cover - network failure fallback
            embedding = None
        memories = self.db.fetch_memories(
            node.id,
            query_embedding=embedding,
            top_k=3,
            min_score=0.1,
        )
        exported: List[Mapping[str, object]] = []
        for item in memories:
            exported.append(
                {
                    "memory_id": item["id"],
                    "content": item["content"],
                    "similarity": item.get("similarity"),
                    "metadata": item.get("metadata", {}),
                }
            )
        return exported

    def _resolve_topic(
        self,
        event: ConversationEvent,
        decision: Mapping[str, Any],
        calltopics: Mapping[str, List[Mapping[str, object]]],
    ) -> tuple[str, str]:
        topic = (decision.get("topic") or "").strip()
        subtopic = (decision.get("subtopic") or "").strip()
        need_recall = bool(decision.get("need_recall"))
        status = decision.get("decision")

        if status == "continue" and topic and subtopic:
            return topic, subtopic

        match_topic = topic
        match_subtopic = subtopic

        if need_recall or not (match_topic and match_subtopic):
            prompt2 = self._call_prompt2(event=event, calltopics=calltopics)
            match = prompt2.get("match") or {}
            if isinstance(match, Mapping) and match.get("topic") and match.get("subtopic"):
                match_topic = str(match["topic"])
                match_subtopic = str(match["subtopic"])
            if prompt2.get("create") or not (match_topic and match_subtopic):
                prompt3 = self._call_prompt3(event=event, calltopics=calltopics)
                match_topic = str(prompt3.get("topic") or match_topic or "新话题").strip() or "新话题"
                match_subtopic = (
                    str(prompt3.get("subtopic") or match_subtopic or "默认子类").strip()
                    or "默认子类"
                )

        if not match_topic:
            match_topic = "未分类"
        if not match_subtopic:
            match_subtopic = "默认子类"
        return match_topic, match_subtopic

    def _sanitize_memory_update_payload(
        self, payload: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        allowed = {
            "topic",
            "subtopic",
            "conv_id",
            "event_index",
            "content",
            "action",
            "recall",
            "metadata",
        }
        sanitized: Dict[str, Any] = {}
        metadata = dict(payload.get("metadata") or {})
        summary = payload.get("summary")
        if summary and not payload.get("content"):
            sanitized["content"] = str(summary)
            metadata.setdefault("event_summary", str(summary))
        for key, value in payload.items():
            if key in allowed and value is not None:
                sanitized[key] = value
        if "content" not in sanitized:
            sanitized["content"] = str(summary or "")
        sanitized["metadata"] = metadata
        return sanitized


@dataclass
class StructuredMemoryAgent:
    """Agent facade that exposes the manager."""

    manager: StructuredMemoryManager

    def run_tempwindow(self, window: ConversationWindow) -> Mapping[str, Any]:
        return self.manager.process_window(window)


__all__ = ["StructuredMemoryAgent", "StructuredMemoryManager"]
