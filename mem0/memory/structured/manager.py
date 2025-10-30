"""High-level orchestration for the structured memory workflow."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from .clients import LocalEmbeddingClient, LocalLLMClient
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
from .schemas import (
    ConversationEvent,
    ConversationWindow,
    MemoryUpdate,
    RecallRequest,
)
from .storage import TopicDatabase
from .tools import SummaryTool, TopicRetrievalTool, TopicUpdateTool

logger = logging.getLogger(__name__)


@dataclass
class StructuredMemoryManager:
    """Manage conversation windows, prompts, and persistence.

    The manager follows the workflow described in the architecture document:

    1. Maintain ``tempwindow`` (conversation window) and caches such as
       ``calltopics`` and ``windowtopics``.
    2. Build and dispatch the four system prompts to a local LLM.
    3. Use tools to retrieve candidate topics, recall historical memories, and
       persist updates.
    4. Store topics / memories in ``TopicDatabase`` with embedding support.
    """

    db: TopicDatabase
    llm_client: LocalLLMClient
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

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------
    def _safe_embed(self, texts):
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
        # produce 8 floats in [0, 1]
        return [int.from_bytes(digest[i : i + 4], "big") / 2**32 for i in range(0, 32, 4)]

    # ------------------------------------------------------------------
    # Conversation ingest helpers
    # ------------------------------------------------------------------
    def add_event(self, *, conv_id: str, role: str, content: str, metadata: Optional[Mapping[str, object]] = None) -> None:
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
        self.window.summary = self.summary_tool(self.window)

    # ------------------------------------------------------------------
    # Prompt execution helpers
    # ------------------------------------------------------------------
    def _call_prompt1(self) -> Mapping[str, object]:
        payload = build_prompt1_payload(
            window=self.window,
            context=self.context,
            calltopics=self._call_topics_payload(),
            windowtopics=self._window_topics_payload(),
        )
        response = self._invoke_llm(payload, SYSTEM_PROMPT_1)
        return self._parse_json_response(response)

    def _call_prompt2(self, recall_requests: List[RecallRequest]) -> Mapping[str, object]:
        raw_events = [
            {"conv_id": event.conv_id, "index": event.index, "content": event.content}
            for event in self.window.events
        ]
        payload = build_prompt2_payload(
            recall_requests=recall_requests,
            calltopics=self._call_topics_payload(),
            windowtopics=self._window_topics_payload(),
            raw_events=raw_events,
        )
        response = self._invoke_llm(payload, SYSTEM_PROMPT_2)
        return self._parse_json_response(response)

    def _call_prompt3(self, tool_results: Mapping[str, object]) -> Mapping[str, object]:
        raw_events = [
            {"conv_id": event.conv_id, "index": event.index, "content": event.content}
            for event in self.window.events
        ]
        payload = build_prompt3_payload(
            tool_results=tool_results,
            calltopics=self._call_topics_payload(),
            windowtopics=self._window_topics_payload(),
            raw_events=raw_events,
        )
        response = self._invoke_llm(payload, SYSTEM_PROMPT_3)
        return self._parse_json_response(response)

    def _call_prompt4(
        self,
        *,
        final_updates: List[MemoryUpdate],
        topic_merges: List[Mapping[str, str]],
        context_updates: List[str],
    ) -> Mapping[str, object]:
        payload = build_prompt4_payload(
            final_updates=final_updates,
            topic_merges=topic_merges,
            context_updates=context_updates,
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
            return {"text": json.dumps({"updates": [], "recall_requests": [], "recent_topics": {}})}
        if system_prompt == SYSTEM_PROMPT_2:
            return {"text": json.dumps({"tool_tasks": [], "context_requirements": []})}
        if system_prompt == SYSTEM_PROMPT_3:
            return {"text": json.dumps({"final_updates": [], "topic_merges": [], "context_updates": []})}
        if system_prompt == SYSTEM_PROMPT_4:
            return {"text": json.dumps({"operations": [], "context_patch": ""})}
        raise ValueError("Unknown system prompt")

    @staticmethod
    def _parse_json_response(response: Mapping[str, object]) -> Mapping[str, object]:
        """Extract the first JSON object from a chat style response."""

        if "choices" in response:
            # Assume OpenAI compatible schema
            message = response["choices"][0]["message"]["content"]
        elif "text" in response:
            message = response["text"]
        else:
            raise ValueError("Local LLM response missing 'choices' or 'text'")

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
            # reset if malformed closing appears
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
        """Run the full prompt pipeline for the current conversation window."""

        # Step 0: build call topics using embeddings
        self.call_topics = {
            topic: [dict(candidate) for candidate in candidates]
            for topic, candidates in self.retrieval_tool(self.window).items()
        }

        window_text = " ".join(event.content for event in self.window.events if event.content)
        window_embedding: Optional[Sequence[float]] = None
        if window_text.strip():
            embeddings = self._safe_embed([window_text])
            window_embedding = embeddings[0] if embeddings else None

        prompt1_result = self._call_prompt1()
        recall_requests = [
            RecallRequest(**payload)
            for payload in prompt1_result.get("recall_requests", [])
        ]
        updates_from_prompt1 = [
            MemoryUpdate(**payload)
            for payload in prompt1_result.get("updates", [])
        ]
        recent_topics = prompt1_result.get("recent_topics", {})
        if isinstance(recent_topics, dict):
            self._merge_window_topics(recent_topics)

        tool_results: Dict[str, object] = {}
        if recall_requests:
            prompt2_result = self._call_prompt2(recall_requests)
            tool_results["prompt2"] = prompt2_result
            for task in prompt2_result.get("tool_tasks", []):
                if task.get("tool") == "topic_recall":
                    topic = task.get("topic")
                    subtopic = task.get("subtopic")
                    node = self.db.fetch_topic(topic, subtopic) if topic and subtopic else None
                    if node:
                        memories = self.db.fetch_memories(
                            node.id,
                            query_embedding=window_embedding,
                            top_k=10,
                            min_score=0.2,
                        )
                    else:
                        memories = []
                    tool_results.setdefault("recall_results", []).append(
                        {
                            "topic": topic,
                            "subtopic": subtopic,
                            "event_summary": node.event_summary if node else "",
                            "memories": memories,
                        }
                    )

        prompt3_result = self._call_prompt3(tool_results)
        final_updates = [
            MemoryUpdate(**payload)
            for payload in prompt3_result.get("final_updates", [])
        ]
        if not final_updates:
            final_updates = updates_from_prompt1

        topic_merges = prompt3_result.get("topic_merges", [])
        context_updates = prompt3_result.get("context_updates", [])

        prompt4_result = self._call_prompt4(
            final_updates=final_updates,
            topic_merges=topic_merges,
            context_updates=context_updates,
        )

        operations = prompt4_result.get("operations", [])
        stored_memory_ids: List[str] = []
        memory_updates: List[MemoryUpdate] = []
        for op in operations:
            if "op" in op and "action" not in op:
                op = {**op}
                op["action"] = op.pop("op")
            memory_update = MemoryUpdate(**op)
            memory_updates.append(memory_update)
            stored_memory_ids.extend(self.update_tool([memory_update]))

        if memory_updates:
            touched: set[Tuple[str, str]] = {
                (update.topic, update.subtopic) for update in memory_updates
            }
            for topic_name, subtopic_name in touched:
                node = self.db.fetch_topic(topic_name, subtopic_name)
                if not node:
                    continue
                topic_entry = self.window_topics.setdefault(topic_name, {})
                entry = topic_entry.setdefault(subtopic_name, {})
                if node.event_summary:
                    entry["summary"] = node.event_summary

        if context_updates:
            self.context["summary"] = "\n".join(context_updates)
        elif prompt4_result.get("context_patch"):
            self.context["summary"] = prompt4_result["context_patch"]

        return {
            "prompt1": prompt1_result,
            "prompt2": tool_results,
            "prompt3": prompt3_result,
            "prompt4": {
                "result": prompt4_result,
                "stored_memory_ids": stored_memory_ids,
            },
        }

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

    def _merge_window_topics(self, recent_topics: Mapping[str, object]) -> None:
        for topic, subtopics in recent_topics.items():
            if not isinstance(subtopics, list):
                continue
            topic_entry = self.window_topics.setdefault(topic, {})
            for item in subtopics:
                if isinstance(item, dict):
                    subtopic = item.get("subtopic") or item.get("name")
                    summary = item.get("summary")
                else:
                    subtopic = str(item)
                    summary = None
                if not subtopic:
                    continue
                entry = topic_entry.setdefault(subtopic, {})
                if summary:
                    entry["summary"] = summary
                elif "summary" not in entry:
                    node = self.db.fetch_topic(topic, subtopic)
                    if node and node.event_summary:
                        entry["summary"] = node.event_summary


@dataclass
class StructuredMemoryAgent:
    """Agent facade that exposes the tools and manager together."""

    manager: StructuredMemoryManager

    def ingest(self, *, conv_id: str, role: str, content: str, metadata: Optional[Mapping[str, object]] = None) -> None:
        self.manager.add_event(conv_id=conv_id, role=role, content=content, metadata=metadata)

    def run(self) -> Mapping[str, object]:
        return self.manager.process_window()


__all__ = ["StructuredMemoryAgent", "StructuredMemoryManager"]
