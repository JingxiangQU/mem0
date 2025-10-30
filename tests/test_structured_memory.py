import json

import pytest

from mem0.memory.structured import (
    LocalEmbeddingClient,
    LocalLLMClient,
    StructuredMemoryAgent,
    StructuredMemoryManager,
)
from mem0.memory.structured.prompts import (
    SYSTEM_PROMPT_1,
    SYSTEM_PROMPT_2,
    SYSTEM_PROMPT_3,
    SYSTEM_PROMPT_4,
)
from mem0.memory.structured.storage import TopicDatabase


def _fake_embedding(texts: list[str]) -> list[list[float]]:
    return [[0.1] * 8 for _ in texts]


def _make_manager(tmp_path):
    db_path = tmp_path / "topics.sqlite"
    database = TopicDatabase(str(db_path))
    manager = StructuredMemoryManager(
        db=database,
        llm_client=LocalLLMClient(),
        embedding_client=LocalEmbeddingClient(),
    )
    manager.embedding_client.embed_texts = _fake_embedding  # type: ignore[method-assign]
    return manager, database


def _default_prompt2_response():
    return {"choices": [{"message": {"content": json.dumps({"recall_tasks": [], "notes": ""})}}]}


def test_structured_memory_pipeline(tmp_path):
    manager, database = _make_manager(tmp_path)

    response_map = {
        SYSTEM_PROMPT_1: iter(
            [
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "decision": {
                                            "topic": "travel",
                                            "subtopic": "japan",
                                            "status": "new_topic",
                                            "need_recall": False,
                                            "create_memory": True,
                                            "reason": "会议中新主题",
                                        },
                                        "summary_clues": ["计划", "日本", "春季出行"],
                                    }
                                )
                            }
                        }
                    ]
                }
            ]
        ),
        SYSTEM_PROMPT_3: iter(
            [
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "final_topic": {
                                            "topic": "travel",
                                            "subtopic": "japan",
                                            "status": "new_topic",
                                            "create_memory": True,
                                            "reason": "首次确认日本行程",
                                        },
                                        "event_summary_focus": "记录旅行动机与时间",
                                        "memory_outline": ["希望春天去日本", "需要制定计划"],
                                        "context_note": "团队开始讨论日本行程。",
                                    }
                                )
                            }
                        }
                    ]
                }
            ]
        ),
        SYSTEM_PROMPT_4: iter(
            [
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "memory_text": "用户计划在春天去日本旅行，需要制定整体安排。",
                                        "event_summary": "日本旅行规划已经启动，参与者确认春季出行目标。",
                                        "context_patch": "正在筹备日本春季旅行。",
                                        "metadata": {"tags": ["travel", "japan"]},
                                    }
                                )
                            }
                        }
                    ]
                }
            ]
        ),
    }

    def fake_generate(*, prompt: str, system_prompt: str, **_: object):  # type: ignore[override]
        iterator = response_map.get(system_prompt)
        if iterator is None:
            if system_prompt == SYSTEM_PROMPT_2:
                return _default_prompt2_response()
            raise AssertionError(f"Unexpected system prompt: {system_prompt}")
        try:
            return next(iterator)
        except StopIteration as exc:  # pragma: no cover - guard for tests
            raise AssertionError(f"No more responses prepared for {system_prompt}") from exc

    manager.llm_client.generate = fake_generate  # type: ignore[method-assign]

    agent = StructuredMemoryAgent(manager=manager)
    agent.ingest(conv_id="conv-1", role="user", content="I want to travel to Japan next spring.")
    result = agent.run()

    assert len(result["events"]) == 1
    event_result = result["events"][0]
    assert event_result["stored_memory_ids"]

    topic_node = database.fetch_topic("travel", "japan")
    assert topic_node is not None
    assert topic_node.event_summary == "日本旅行规划已经启动，参与者确认春季出行目标。"

    memories = database.fetch_memories(topic_node.id)
    assert len(memories) == 1
    assert memories[0]["content"] == "用户计划在春天去日本旅行，需要制定整体安排。"
    assert manager.context["summary"] == "正在筹备日本春季旅行。"
    assert manager.window_topics["travel"]["japan"]["summary"] == "日本旅行规划已经启动，参与者确认春季出行目标。"


def test_event_continuation_updates_existing_summary(tmp_path):
    manager, database = _make_manager(tmp_path)

    database.upsert_topic(
        topic="travel",
        subtopic="japan",
        embedding=[0.1] * 8,
        event_summary="日本旅行初步规划已完成。",
    )

    response_map = {
        SYSTEM_PROMPT_1: iter(
            [
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "decision": {
                                            "topic": "travel",
                                            "subtopic": "japan",
                                            "status": "continuation",
                                            "need_recall": False,
                                            "create_memory": True,
                                            "reason": "围绕同一旅行主题追加酒店信息",
                                        },
                                        "summary_clues": ["酒店", "东京"],
                                    }
                                )
                            }
                        }
                    ]
                }
            ]
        ),
        SYSTEM_PROMPT_3: iter(
            [
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "final_topic": {
                                            "topic": "travel",
                                            "subtopic": "japan",
                                            "status": "continuation",
                                            "create_memory": True,
                                            "reason": "确认行程细化",
                                        },
                                        "event_summary_focus": "总结酒店选择进展",
                                        "memory_outline": ["讨论东京酒店", "继续推进日本旅行"],
                                        "context_note": "日本旅行计划进入酒店筛选阶段。",
                                    }
                                )
                            }
                        }
                    ]
                }
            ]
        ),
        SYSTEM_PROMPT_4: iter(
            [
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "memory_text": "团队开始研究东京酒店，确认预算与地段。",
                                        "event_summary": "日本旅行规划继续推进，已经讨论东京酒店预订。",
                                        "context_patch": "日本旅行计划正在确定东京住宿方案。",
                                        "metadata": {"tags": ["hotel"]},
                                    }
                                )
                            }
                        }
                    ]
                }
            ]
        ),
    }

    def fake_generate(*, prompt: str, system_prompt: str, **_: object):  # type: ignore[override]
        iterator = response_map.get(system_prompt)
        if iterator is None:
            return _default_prompt2_response()
        try:
            return next(iterator)
        except StopIteration as exc:  # pragma: no cover
            raise AssertionError(f"No more responses prepared for {system_prompt}") from exc

    manager.llm_client.generate = fake_generate  # type: ignore[method-assign]

    agent = StructuredMemoryAgent(manager=manager)
    agent.ingest(conv_id="conv-2", role="user", content="What about hotels in Tokyo?")
    result = agent.run()

    assert result["events"][0]["stored_memory_ids"]
    node = database.fetch_topic("travel", "japan")
    assert node.event_summary == "日本旅行规划继续推进，已经讨论东京酒店预订。"
    memories = database.fetch_memories(node.id)
    assert memories[-1]["metadata"]["status"] == "continuation"


def test_topic_recall_branch_flow(tmp_path):
    manager, database = _make_manager(tmp_path)

    topic_id = database.upsert_topic(
        topic="finance",
        subtopic="budget",
        embedding=[0.1] * 8,
        event_summary="年度预算已建立，需要跟踪执行。",
    )
    database.add_memory(
        topic_id=topic_id,
        conv_id="conv-prev",
        event_index=0,
        content="预算会会议纪要。",
        embedding=[0.1] * 8,
        metadata={"action": "add"},
    )

    response_map = {
        SYSTEM_PROMPT_1: iter(
            [
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "decision": {
                                            "topic": "finance",
                                            "subtopic": "budget",
                                            "status": "continuation",
                                            "need_recall": True,
                                            "create_memory": True,
                                            "reason": "需要回顾之前的预算讨论",
                                        },
                                        "summary_clues": ["预算执行", "复盘"],
                                    }
                                )
                            }
                        }
                    ]
                }
            ]
        ),
        SYSTEM_PROMPT_2: iter(
            [
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "recall_tasks": [
                                            {
                                                "topic": "finance",
                                                "subtopic": "budget",
                                                "query": "年度预算跟进",
                                            }
                                        ],
                                        "notes": "回顾上一季度预算事项",
                                    }
                                )
                            }
                        }
                    ]
                }
            ]
        ),
        SYSTEM_PROMPT_3: iter(
            [
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "final_topic": {
                                            "topic": "finance",
                                            "subtopic": "budget",
                                            "status": "continuation",
                                            "create_memory": True,
                                            "reason": "确认预算复盘",
                                        },
                                        "event_summary_focus": "补充预算执行状态",
                                        "memory_outline": ["复盘预算执行", "识别风险"],
                                        "context_note": "预算会议要求跟踪执行风险。",
                                    }
                                )
                            }
                        }
                    ]
                }
            ]
        ),
        SYSTEM_PROMPT_4: iter(
            [
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "memory_text": "团队复盘年度预算执行情况，识别新的风险点。",
                                        "event_summary": "预算主题持续推进，目前关注执行风险。",
                                        "context_patch": "持续跟踪预算执行进度。",
                                        "metadata": {"tags": ["finance"]},
                                    }
                                )
                            }
                        }
                    ]
                }
            ]
        ),
    }

    def fake_generate(*, prompt: str, system_prompt: str, **_: object):  # type: ignore[override]
        iterator = response_map.get(system_prompt)
        if iterator is None:
            return _default_prompt2_response()
        try:
            return next(iterator)
        except StopIteration as exc:  # pragma: no cover - guard
            raise AssertionError(f"No more responses prepared for {system_prompt}") from exc

    manager.llm_client.generate = fake_generate  # type: ignore[method-assign]

    agent = StructuredMemoryAgent(manager=manager)
    agent.ingest(conv_id="conv-3", role="assistant", content="Let's review the annual budget risks.")
    result = agent.run()

    assert result["events"][0]["prompts"]["prompt2"]["notes"] == "回顾上一季度预算事项"
    assert manager.context["summary"] == "持续跟踪预算执行进度。"


def test_multi_event_window_processing(tmp_path):
    manager, database = _make_manager(tmp_path)

    responses_p1 = iter(
        [
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "decision": {
                                        "topic": "project",
                                        "subtopic": "alpha",
                                        "status": "new_topic",
                                        "need_recall": False,
                                        "create_memory": True,
                                        "reason": "启动项目 Alpha",
                                    },
                                    "summary_clues": ["Alpha", "目标"],
                                }
                            )
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "decision": {
                                        "topic": "project",
                                        "subtopic": "beta",
                                        "status": "new_topic",
                                        "need_recall": False,
                                        "create_memory": True,
                                        "reason": "切换到 Beta 讨论",
                                    },
                                    "summary_clues": ["Beta"],
                                }
                            )
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "decision": {
                                        "topic": "project",
                                        "subtopic": "alpha",
                                        "status": "continuation",
                                        "need_recall": False,
                                        "create_memory": True,
                                        "reason": "回到 Alpha 任务",
                                    },
                                    "summary_clues": ["进度"],
                                }
                            )
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "decision": {
                                        "topic": "project",
                                        "subtopic": "gamma",
                                        "status": "new_topic",
                                        "need_recall": False,
                                        "create_memory": True,
                                        "reason": "新建 Gamma 话题",
                                    },
                                    "summary_clues": ["Gamma"],
                                }
                            )
                        }
                    }
                ]
            },
        ]
    )

    def multi_iter(template):
        return iter([
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(template(i))
                        }
                    }
                ]
            }
            for i in range(4)
        ])

    def template3(idx):
        topics = [
            {
                "topic": "project",
                "subtopic": sub,
                "status": status,
                "create_memory": True,
                "reason": f"事件 {idx}"
            }
            for sub, status in zip(["alpha", "beta", "alpha", "gamma"], ["new_topic", "new_topic", "continuation", "new_topic"])
        ]
        focus = [
            "Alpha 目标",
            "Beta 讨论要点",
            "Alpha 进度",
            "Gamma 新计划",
        ]
        context = [
            "Alpha 项目启动",
            "Beta 项目讨论展开",
            "Alpha 项目推进",
            "Gamma 计划建立",
        ]
        outlines = [
            ["设定 Alpha 目标"],
            ["确认 Beta 流程"],
            ["回顾 Alpha 进展"],
            ["定义 Gamma 方案"],
        ]
        return {
            "final_topic": topics[idx],
            "event_summary_focus": focus[idx],
            "memory_outline": outlines[idx],
            "context_note": context[idx],
        }

    def template4(idx):
        summaries = [
            "Alpha 项目正式启动。",
            "Beta 项目框架成形。",
            "Alpha 项目完成阶段评估。",
            "Gamma 项目完成启动准备。",
        ]
        memories = [
            "讨论 Alpha 项目目标。",
            "记录 Beta 项目讨论要点。",
            "总结 Alpha 项目进展。",
            "建立 Gamma 项目计划。",
        ]
        contexts = [
            "Alpha 项目计划制定中。",
            "Beta 项目讨论中。",
            "Alpha 项目推进中。",
            "Gamma 项目准备中。",
        ]
        return {
            "memory_text": memories[idx],
            "event_summary": summaries[idx],
            "context_patch": contexts[idx],
            "metadata": {"tags": [f"project-{idx}"]},
        }

    response_map = {
        SYSTEM_PROMPT_1: responses_p1,
        SYSTEM_PROMPT_3: multi_iter(template3),
        SYSTEM_PROMPT_4: multi_iter(template4),
    }

    def fake_generate(*, prompt: str, system_prompt: str, **_: object):  # type: ignore[override]
        iterator = response_map.get(system_prompt)
        if iterator is None:
            return _default_prompt2_response()
        try:
            return next(iterator)
        except StopIteration as exc:  # pragma: no cover
            raise AssertionError(f"No more responses prepared for {system_prompt}") from exc

    manager.llm_client.generate = fake_generate  # type: ignore[method-assign]

    agent = StructuredMemoryAgent(manager=manager)
    agent.ingest(conv_id="conv-4", role="user", content="Alpha kickoff.")
    agent.ingest(conv_id="conv-4", role="assistant", content="Switching to beta requirements.")
    agent.ingest(conv_id="conv-4", role="user", content="Alpha progress update.")
    agent.ingest(conv_id="conv-4", role="assistant", content="Introduce gamma initiative.")
    result = agent.run()

    assert len(result["events"]) == 4
    summaries = {
        (topic, sub): info["summary"]
        for topic, subtopics in manager.window_topics.items()
        for sub, info in subtopics.items()
    }
    assert summaries[("project", "alpha")] == "Alpha 项目完成阶段评估。"
    assert manager.context["summary"] == "Gamma 项目准备中。"
