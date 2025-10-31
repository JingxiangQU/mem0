from __future__ import annotations

import json
from typing import Any, Iterable, List, Mapping, Sequence

from mem0.memory.structured.manager import StructuredMemoryManager
from mem0.memory.structured.prompts import (
    SYSTEM_PROMPT_1,
    SYSTEM_PROMPT_2,
    SYSTEM_PROMPT_3,
    SYSTEM_PROMPT_4,
    TOPIC_EVENT0_PROMPT,
)
from mem0.memory.structured.schemas import ConversationEvent, ConversationWindow
from mem0.memory.structured.storage import TopicDatabase


class FakeLLMClient:
    def __init__(self, responses: Mapping[str, List[str]]) -> None:
        self.responses = {key: list(queue) for key, queue in responses.items()}
        self.calls: List[Mapping[str, Any]] = []

    def chat(
        self, messages: Sequence[Mapping[str, Any]], *, extra_body: Mapping[str, Any] | None = None
    ) -> str:
        system_prompt = next(msg["content"] for msg in messages if msg["role"] == "system")
        queue = self.responses.get(system_prompt)
        if not queue:
            raise AssertionError(f"No response queued for system prompt: {system_prompt!r}")
        reply = queue.pop(0)
        self.calls.append({"system": system_prompt, "messages": list(messages)})
        return reply

    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for idx, text in enumerate(texts):
            base = float(len(text) + idx + 1)
            vectors.append([base, base / 10.0, base / 100.0])
        return vectors


class FakeEmbeddingClient:
    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        return [[float(idx + 1)] * 3 for idx, _ in enumerate(texts)]


def _make_manager(responses: Mapping[str, List[str]]) -> tuple[StructuredMemoryManager, TopicDatabase, FakeLLMClient]:
    db = TopicDatabase(":memory:")
    llm = FakeLLMClient(responses)
    embed = FakeEmbeddingClient()
    manager = StructuredMemoryManager(db=db, llm_client=llm, embedding_client=embed)
    return manager, db, llm


def _window(conv_id: str, contents: Sequence[str]) -> ConversationWindow:
    events = [
        ConversationEvent(conv_id=conv_id, index=idx, role="user", content=text)
        for idx, text in enumerate(contents)
    ]
    return ConversationWindow(window_id=conv_id, events=events)


def _queue_json(data: Mapping[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False)


def test_event_continuation_updates_existing_summary() -> None:
    responses = {
        SYSTEM_PROMPT_1: [
            _queue_json(
                {
                    "decision": "continue",
                    "topic": "travel",
                    "subtopic": "japan",
                    "summary": None,
                    "need_recall": False,
                }
            )
        ],
        SYSTEM_PROMPT_4: [_queue_json({"summary": "讨论东京酒店方案。"})],
        TOPIC_EVENT0_PROMPT: ["历史摘要结合新对话，形成最新东京行程总结。"],
    }

    manager, db, _ = _make_manager(responses)
    db.upsert_topic(topic="travel", subtopic="japan", event_summary="已有东京行程摘要")

    window = _window("travel::session", ["What about hotels in Tokyo?"])
    result = manager.process_window(window)

    assert result["events"][0]["topic"] == "travel"
    assert result["events"][0]["subtopic"] == "japan"
    assert result["events"][0]["summary"] == "讨论东京酒店方案。"

    topic_node = db.fetch_topic("travel", "japan")
    assert topic_node is not None
    assert topic_node.event_summary == "历史摘要结合新对话，形成最新东京行程总结。"

    cur = db.connection.execute("SELECT conv_id, event_index, content FROM memories")
    row = cur.fetchone()
    assert row["conv_id"] == "travel::session"
    assert row["event_index"] == 0
    assert row["content"] == "讨论东京酒店方案。"


def test_topic_recall_branch_flow() -> None:
    responses = {
        SYSTEM_PROMPT_1: [
            _queue_json(
                {
                    "decision": "uncertain",
                    "topic": None,
                    "subtopic": None,
                    "summary": None,
                    "need_recall": True,
                }
            )
        ],
        SYSTEM_PROMPT_2: [_queue_json({"match": {"topic": "project", "subtopic": "roadmap"}, "create": False})],
        SYSTEM_PROMPT_4: [_queue_json({"summary": "确认产品路线图里程碑。"})],
        TOPIC_EVENT0_PROMPT: ["产品路线图新增一个里程碑要点。"],
    }

    manager, db, _ = _make_manager(responses)
    db.upsert_topic(topic="project", subtopic="roadmap", event_summary="原有路线图摘要")

    manager.retrieval_tool = lambda window: {  # type: ignore[assignment]
        "project": [{"topic": "project", "subtopic": "roadmap", "summary": "原有路线图摘要"}]
    }

    window = _window("meeting-1", ["We should add a beta milestone next month."])
    result = manager.process_window(window)

    event_result = result["events"][0]
    assert event_result["topic"] == "project"
    assert event_result["subtopic"] == "roadmap"
    assert event_result["summary"] == "确认产品路线图里程碑。"

    topic_node = db.fetch_topic("project", "roadmap")
    assert topic_node.event_summary == "产品路线图新增一个里程碑要点。"


def test_multi_event_window_processing() -> None:
    responses = {
        SYSTEM_PROMPT_1: [
            _queue_json(
                {
                    "decision": "new_event",
                    "topic": "marketing",
                    "subtopic": "launch",
                    "summary": "规划发布会流程。",
                    "need_recall": False,
                }
            ),
            _queue_json(
                {
                    "decision": "new_event",
                    "topic": "sales",
                    "subtopic": "training",
                    "summary": "安排销售培训。",
                    "need_recall": False,
                }
            ),
            _queue_json(
                {
                    "decision": "continue",
                    "topic": "marketing",
                    "subtopic": "launch",
                    "summary": None,
                    "need_recall": False,
                }
            ),
            _queue_json(
                {
                    "decision": "new_event",
                    "topic": "support",
                    "subtopic": "workflow",
                    "summary": "制定客服响应流程。",
                    "need_recall": False,
                }
            ),
        ],
        SYSTEM_PROMPT_4: [
            _queue_json({"summary": "确定发布会时间与职责。"}),
            _queue_json({"summary": "确定销售培训时间表。"}),
            _queue_json({"summary": "补充发布会宣传要求。"}),
            _queue_json({"summary": "建立客服工单分配。"}),
        ],
        TOPIC_EVENT0_PROMPT: [
            "发布会整体安排包括时间与职责。",
            "销售培训覆盖时间表。",
            "发布会摘要加入宣传要求。",
            "客服流程摘要形成。",
        ],
    }

    manager, db, _ = _make_manager(responses)
    manager.retrieval_tool = lambda window: {}  # type: ignore[assignment]

    window = _window(
        "meeting-window",
        [
            "Let's confirm the launch event details.",
            "Schedule the sales training for next week.",
            "We also need launch marketing materials.",
            "Customer support must have a new workflow.",
        ],
    )
    result = manager.process_window(window)

    topics = {(item["topic"], item["subtopic"]) for item in result["events"]}
    assert topics == {
        ("marketing", "launch"),
        ("sales", "training"),
        ("support", "workflow"),
    }

    marketing_events = [item for item in result["events"] if item["topic"] == "marketing"]
    assert len(marketing_events) == 2

    cur = db.connection.execute("SELECT topic_id, conv_id, event_index FROM memories ORDER BY event_index")
    rows = cur.fetchall()
    assert len(rows) == 4
    assert [row["event_index"] for row in rows] == [0, 1, 2, 3]

