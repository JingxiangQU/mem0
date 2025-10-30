"""High-level orchestration for the structured memory workflow."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from .clients import ExternalLLMClient, LocalEmbeddingClient, LocalLLMClient
from .prompts import (
    SYSTEM_PROMPT_1,
    SYSTEM_PROMPT_2,
    SYSTEM_PROMPT_3,
    SYSTEM_PROMPT_4,
    build_prompt1_payload,
    build_prompt2_payload,
    build_prompt3_payload,
    build_prompt4_payload,
)
from .schemas import ConversationEvent, ConversationWindow, MemoryUpdate
from .storage import TopicDatabase
from .tools import SummaryTool, TopicRetrievalTool, TopicUpdateTool

logger = logging.getLogger(__name__)


@dataclass
class StructuredMemoryManager:
    """Manage conversation windows, prompts, and persistence."""

    db: TopicDatabase
    llm_client: LocalLLMClient | ExternalLLMClient
    embedding_client: LocalEmbeddingClient
    window_size: int = 6
    context: MutableMapping[str, str] = field(default_factory=dict)
    window_topics: MutableMapping[str, Dict[str, Dict[str, object]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.window = ConversationWindow(window_id="window-0")
        self.call_topics: MutableMapping[str, List[Mapping[str, object]]] = {}
        self.summary_tool = SummaryTool()
        self.retrieval_tool = TopicRetrievalTool(
            db=self.db,
            embed_texts=self._safe_embed,
        )
        self.update_tool = TopicUpdateTool(
            db=self.db,
            embed_texts=self._safe_embed,
        )
        self._processed_count = 0
        self._event_counter = 0

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------
    def _safe_embed(self, texts: Iterable[str]):
        texts = list(texts)
        if not texts:
            return []
        try:
            return self.embedding_client.embed_texts(texts)
        except Exception as exc:  # pragma: no cover - network failure fallback
            logger.warning("Embedding service unavailable, falling back to hash embeddings: %s", exc)
            return [self._hash_embedding(text) for text in texts]

    @staticmethod
    def _hash_embedding(text: str) -> List[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [int.from_bytes(digest[i : i + 4], "big") / 2**32 for i in range(0, 32, 4)]

    # ------------------------------------------------------------------
    # Conversation ingest helpers
    # ------------------------------------------------------------------
    def add_event(
        self,
        *,
        conv_id: str,
        role: str,
        content: str,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> None:
        index = len(self.window.events)
        event = ConversationEvent(
            conv_id=conv_id,
            index=index,
            role=role,
            content=content,
            metadata=dict(metadata or {}),
        )
        self.window.append(event)
        if len(self.window.events) > self.window_size:
            self.window.events = self.window.events[-self.window_size :]
            self._processed_count = max(0, self._processed_count - 1)
        self.window.summary = self.summary_tool(self.window)

    # ------------------------------------------------------------------
    # Prompt execution helpers
    # ------------------------------------------------------------------
    def _call_prompt1(
        self,
        *,
        event: ConversationEvent,
        history: Sequence[ConversationEvent],
        calltopics: Mapping[str, list[Mapping[str, object]]],
    ) -> Mapping[str, object]:
        payload = build_prompt1_payload(
            context_summary=self.context.get("summary"),
            window_id=self.window.window_id,
            position=self._event_counter + 1,
            event=event,
            history=history,
            calltopics=calltopics,
            windowtopics=self._window_topics_payload(),
            topic_summaries=self._topic_summary_snapshots(),
        )
        response = self._invoke_llm(payload, SYSTEM_PROMPT_1)
        return self._parse_json_response(response)

    def _call_prompt2(
        self,
        *,
        decision: Mapping[str, object],
        event: ConversationEvent,
        calltopics: Mapping[str, list[Mapping[str, object]]],
    ) -> Mapping[str, object]:
        payload = build_prompt2_payload(
            decision=decision,
            event=event,
            calltopics=calltopics,
            windowtopics=self._window_topics_payload(),
        )
        response = self._invoke_llm(payload, SYSTEM_PROMPT_2)
        return self._parse_json_response(response)

    def _call_prompt3(
        self,
        *,
        decision: Mapping[str, object],
        event: ConversationEvent,
        recall_results: Sequence[Mapping[str, object]],
        calltopics: Mapping[str, list[Mapping[str, object]]],
    ) -> Mapping[str, object]:
        payload = build_prompt3_payload(
            decision=decision,
            event=event,
            recall_results=recall_results,
            calltopics=calltopics,
            windowtopics=self._window_topics_payload(),
        )
        response = self._invoke_llm(payload, SYSTEM_PROMPT_3)
        return self._parse_json_response(response)

    def _call_prompt4(
        self,
        *,
        final_topic: Mapping[str, object],
        event: ConversationEvent,
        history_payload: Sequence[Mapping[str, object]],
        previous_summary: str | None,
        summary_focus: str | None,
        memory_outline: Sequence[str] | None,
        context_note: str | None,
    ) -> Mapping[str, object]:
        payload = build_prompt4_payload(
            final_topic=final_topic,
            event=event,
            history=history_payload,
            previous_summary=previous_summary,
            summary_focus=summary_focus,
            memory_outline=memory_outline,
            context_note=context_note,
        )
        response = self._invoke_llm(payload, SYSTEM_PROMPT_4)
        return self._parse_json_response(response)

    def _invoke_llm(self, payload: str, system_prompt: str) -> Mapping[str, object]:
        try:
            return self.llm_client.generate(prompt=payload, system_prompt=system_prompt)
        except Exception as exc:  # pragma: no cover - network failure fallback
            logger.warning("LLM service unavailable, returning fallback response: %s", exc)
            return self._fallback_response(system_prompt)

    def _fallback_response(self, system_prompt: str) -> Mapping[str, object]:
        if system_prompt == SYSTEM_PROMPT_1:
            return {
                "text": json.dumps(
                    {
                        "decision": {
                            "topic": "",
                            "subtopic": "",
                            "status": "new_topic",
                            "need_recall": False,
                            "create_memory": False,
                            "reason": "fallback",
                        },
                        "summary_clues": [],
                    }
                )
            }
        if system_prompt == SYSTEM_PROMPT_2:
            return {"text": json.dumps({"recall_tasks": [], "notes": ""})}
        if system_prompt == SYSTEM_PROMPT_3:
            return {
                "text": json.dumps(
                    {
                        "final_topic": {
                            "topic": "",
                            "subtopic": "",
                            "status": "new_topic",
                            "create_memory": False,
                            "reason": "fallback",
                        },
                        "event_summary_focus": "",
                        "memory_outline": [],
                        "context_note": "",
                    }
                )
            }
        if system_prompt == SYSTEM_PROMPT_4:
            return {
                "text": json.dumps(
                    {
                        "memory_text": "",
                        "event_summary": "",
                        "context_patch": self.context.get("summary", ""),
                        "metadata": {},
                    }
                )
            }
        raise ValueError("Unknown system prompt")

    @staticmethod
    def _parse_json_response(response: Mapping[str, object]) -> Mapping[str, object]:
        if "choices" in response:
            message = response["choices"][0]["message"]["content"]
        elif "text" in response:
            message = response["text"]
        else:
            raise ValueError("LLM response missing 'choices' or 'text'")

        parsed = StructuredMemoryManager._coerce_json_blob(message)
        if parsed is None:
            raise ValueError("Unable to extract JSON object from LLM response")
        return parsed

    @staticmethod
    def _coerce_json_blob(message: str) -> Optional[Mapping[str, object]]:
        sanitized = message.strip()
        if sanitized.startswith("```"):
            sanitized = sanitized[3:]
            if sanitized.lower().startswith("json"):
                sanitized = sanitized[4:]
            sanitized = sanitized.lstrip("\n")
            if sanitized.endswith("```"):
                sanitized = sanitized[:-3]
        elif sanitized.lower().startswith("json"):
            sanitized = sanitized[4:]
            sanitized = sanitized.lstrip(": ")

        try:
            return json.loads(sanitized)
        except json.JSONDecodeError:
            pass

        blob = StructuredMemoryManager._first_json_object(sanitized)
        if blob is None:
            return None
        return json.loads(blob)

    @staticmethod
    def _first_json_object(message: str) -> Optional[str]:
        start: Optional[int] = None
        depth = 0
        in_string = False
        escape = False
        for idx, char in enumerate(message):
            if start is None:
                if char == "{":
                    start = idx
                    depth = 1
                    in_string = False
                    escape = False
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
                            return message[start : idx + 1]
            if depth < 0:
                start = None
                depth = 0
                in_string = False
                escape = False
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process_window(self) -> Mapping[str, object]:
        new_events = self.window.events[self._processed_count :]
        if not new_events:
            return {"events": [], "context": dict(self.context)}

        event_results: List[Mapping[str, object]] = []
        for event in new_events:
            history = self._recent_history(event.index)
            single_window = ConversationWindow(window_id=self.window.window_id, events=[event])
            calltopics = self.retrieval_tool(single_window)
            self.call_topics = calltopics

            prompt1_result = self._call_prompt1(
                event=event,
                history=history,
                calltopics=self._call_topics_payload(),
            )
            decision = prompt1_result.get("decision", {})
            summary_clues = prompt1_result.get("summary_clues", [])

            prompt2_result: Mapping[str, object] = {"recall_tasks": [], "notes": ""}
            recall_results: List[Mapping[str, object]] = []
            if decision.get("need_recall"):
                prompt2_result = self._call_prompt2(
                    decision=decision,
                    event=event,
                    calltopics=self._call_topics_payload(),
                )
                for task in prompt2_result.get("recall_tasks", []):
                    topic = task.get("topic")
                    subtopic = task.get("subtopic")
                    if not topic:
                        continue
                    node = self.db.fetch_topic(topic, subtopic) if subtopic else None
                    if node is None and subtopic:
                        continue
                    query_embedding = None
                    query_text = task.get("query")
                    if isinstance(query_text, str) and query_text.strip():
                        embeddings = self._safe_embed([query_text])
                        query_embedding = embeddings[0] if embeddings else None
                    if node:
                        memories = self.db.fetch_memories(
                            node.id,
                            query_embedding=query_embedding,
                            top_k=10,
                            min_score=0.2 if query_embedding is not None else None,
                        )
                        summary = node.event_summary or ""
                    else:
                        memories = []
                        summary = ""
                    recall_results.append(
                        {
                            "topic": topic,
                            "subtopic": subtopic or "",
                            "event_summary": summary,
                            "memories": memories,
                        }
                    )

            prompt3_result = self._call_prompt3(
                decision=decision,
                event=event,
                recall_results=recall_results,
                calltopics=self._call_topics_payload(),
            )

            final_topic = prompt3_result.get("final_topic") or decision
            topic_name = (final_topic or {}).get("topic") or decision.get("topic")
            subtopic_name = (final_topic or {}).get("subtopic") or decision.get("subtopic")
            if not topic_name or not subtopic_name:
                logger.debug("Skipping event %s due to missing topic/subtopic", event.index)
                self._event_counter += 1
                event_results.append(
                    {
                        "event_index": event.index,
                        "decision": decision,
                        "prompts": {
                            "prompt1": prompt1_result,
                            "prompt2": prompt2_result,
                            "prompt3": prompt3_result,
                            "prompt4": {},
                        },
                        "stored_memory_ids": [],
                    }
                )
                continue

            status = final_topic.get("status") or decision.get("status") or "new_topic"
            create_memory = bool(final_topic.get("create_memory", True))
            reason = final_topic.get("reason") or decision.get("reason")
            summary_focus = prompt3_result.get("event_summary_focus")
            memory_outline = prompt3_result.get("memory_outline")
            context_note = prompt3_result.get("context_note") or ""

            prompt4_result = self._call_prompt4(
                final_topic=final_topic,
                event=event,
                history_payload=[item.to_payload() for item in history],
                previous_summary=self._topic_summary(topic_name, subtopic_name),
                summary_focus=summary_focus,
                memory_outline=memory_outline,
                context_note=context_note,
            )

            metadata = dict(prompt4_result.get("metadata") or {})
            metadata.setdefault("status", status)
            if reason:
                metadata.setdefault("reason", reason)
            if summary_clues:
                metadata.setdefault("summary_clues", summary_clues)
            metadata["tempwindow_index"] = self._event_counter + 1

            memory_text = prompt4_result.get("memory_text") or ""
            event_summary_text = prompt4_result.get("event_summary") or ""
            context_patch = prompt4_result.get("context_patch") or context_note

            memory_updates: List[MemoryUpdate] = []
            summary_overrides: Dict[tuple[str, str], str] = {}
            if create_memory and memory_text.strip():
                update = MemoryUpdate(
                    topic=topic_name,
                    subtopic=subtopic_name,
                    conv_id=event.conv_id,
                    event_index=event.index,
                    content=memory_text.strip(),
                    action="update" if status == "continuation" else "add",
                    metadata=metadata,
                )
                memory_updates.append(update)
            if event_summary_text.strip():
                summary_overrides[(topic_name, subtopic_name)] = event_summary_text.strip()

            stored_memory_ids: List[str] = []
            if memory_updates or summary_overrides:
                stored_memory_ids = self.update_tool(
                    memory_updates,
                    summary_overrides=summary_overrides or None,
                )

            if summary_overrides:
                for (topic_label, subtopic_label), summary_text in summary_overrides.items():
                    topic_entry = self.window_topics.setdefault(topic_label, {})
                    entry = topic_entry.setdefault(subtopic_label, {})
                    entry["summary"] = summary_text

            if context_patch:
                self.context["summary"] = context_patch

            event_results.append(
                {
                    "event_index": event.index,
                    "decision": decision,
                    "prompts": {
                        "prompt1": prompt1_result,
                        "prompt2": prompt2_result,
                        "prompt3": prompt3_result,
                        "prompt4": prompt4_result,
                    },
                    "stored_memory_ids": stored_memory_ids,
                }
            )

            self._event_counter += 1

        self._processed_count += len(new_events)
        return {"events": event_results, "context": dict(self.context)}

    # ------------------------------------------------------------------
    # Topic cache helpers
    # ------------------------------------------------------------------
    def _call_topics_payload(self) -> Dict[str, List[Mapping[str, object]]]:
        payload: Dict[str, List[Mapping[str, object]]] = {}
        for topic, candidates in self.call_topics.items():
            payload[topic] = [dict(candidate) for candidate in candidates]
        return payload

    def _window_topics_payload(self) -> Dict[str, List[Mapping[str, object]]]:
        payload: Dict[str, List[Mapping[str, object]]] = {}
        for topic, subtopics in self.window_topics.items():
            entries = [
                {
                    "topic": topic,
                    "subtopic": name,
                    **{key: value for key, value in details.items()},
                }
                for name, details in subtopics.items()
            ]
            entries.sort(key=lambda item: item.get("subtopic", ""))
            payload[topic] = entries
        return payload

    def _topic_summary_snapshots(self) -> List[Mapping[str, str]]:
        snapshots: List[Mapping[str, str]] = []
        seen: set[Tuple[str, str]] = set()
        for topic, subtopics in self.window_topics.items():
            for name, details in subtopics.items():
                summary = details.get("summary")
                if summary:
                    snapshots.append({"topic": topic, "subtopic": name, "event_summary": summary})
                    seen.add((topic, name))
        for topic, candidates in self.call_topics.items():
            for candidate in candidates:
                subtopic = candidate.get("subtopic")
                if not subtopic:
                    continue
                key = (topic, subtopic)
                if key in seen:
                    continue
                node = self.db.fetch_topic(topic, subtopic)
                if node and node.event_summary:
                    snapshots.append(
                        {
                            "topic": topic,
                            "subtopic": subtopic,
                            "event_summary": node.event_summary,
                        }
                    )
                    seen.add(key)
        return snapshots

    def _topic_summary(self, topic: str, subtopic: str) -> Optional[str]:
        topic_entry = self.window_topics.get(topic, {}).get(subtopic, {})
        summary = topic_entry.get("summary")
        if summary:
            return summary
        node = self.db.fetch_topic(topic, subtopic)
        return node.event_summary if node else None

    def _recent_history(self, index: int, lookback: int = 3) -> Sequence[ConversationEvent]:
        start = max(0, index - lookback)
        return self.window.events[start:index]


@dataclass
class StructuredMemoryAgent:
    """Agent facade that exposes the tools and manager together."""

    manager: StructuredMemoryManager

    def ingest(
        self,
        *,
        conv_id: str,
        role: str,
        content: str,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> None:
        self.manager.add_event(conv_id=conv_id, role=role, content=content, metadata=metadata)

    def run(self) -> Mapping[str, object]:
        return self.manager.process_window()


__all__ = ["StructuredMemoryAgent", "StructuredMemoryManager"]
