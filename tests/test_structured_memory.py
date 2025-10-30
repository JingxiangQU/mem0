from __future__ import annotations

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


def test_structured_memory_pipeline(tmp_path):
    manager, database = _make_manager(tmp_path)

    responses = iter(
        [
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "updates": [
                                        {
                                            "topic": "travel",
                                            "subtopic": "japan",
                                            "conv_id": "conv-1",
                                            "event_index": 0,
                                            "content": "User wants to visit Japan",
                                            "action": "add",
                                            "metadata": {"reason": "new_topic"},
                                        }
                                    ],
                                    "recall_requests": [],
                                    "recent_topics": {
                                        "travel": [
                                            {
                                                "subtopic": "japan",
                                                "summary": "用户计划去日本旅行",
                                            }
                                        ]
                                    },
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
                                    "final_updates": [],
                                    "topic_merges": [],
                                    "context_updates": ["用户计划去日本旅行"],
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
                                    "operations": [
                                        {
                                            "action": "add",
                                            "topic": "travel",
                                            "subtopic": "japan",
                                            "conv_id": "conv-1",
                                            "event_index": 0,
                                            "content": "User wants to visit Japan",
                                            "metadata": {"source": "new_topic"},
                                        }
                                    ],
                                    "context_patch": "用户计划去日本旅行",
                                }
                            )
                        }
                    }
                ]
            },
        ]
    )

    def fake_generate(*, prompt: str, system_prompt: str):  # type: ignore[override]
        return next(responses)

    manager.llm_client.generate = fake_generate  # type: ignore[method-assign]

    agent = StructuredMemoryAgent(manager=manager)
    agent.ingest(conv_id="conv-1", role="user", content="I want to travel to Japan next spring.")
    result = agent.run()

    stored_ids = result["prompt4"]["stored_memory_ids"]
    assert len(stored_ids) == 1

    topic_node = database.fetch_topic("travel", "japan")
    assert topic_node is not None
    assert topic_node.event_summary == "User wants to visit Japan"
    memories = database.fetch_memories(topic_node.id)
    assert memories
    assert memories[0]["content"] == "User wants to visit Japan"
    assert memories[0]["similarity"] is None

    filtered = database.fetch_memories(topic_node.id, query_embedding=[0.1] * 8)
    assert filtered
    assert filtered[0]["similarity"] == pytest.approx(1.0)

    assert manager.context["summary"] == "用户计划去日本旅行"
    assert manager.window_topics["travel"]["japan"]["summary"] == "User wants to visit Japan"


def test_event_continuation_updates_existing_summary(tmp_path):
    manager, database = _make_manager(tmp_path)

    database.upsert_topic(
        topic="travel",
        subtopic="japan",
        embedding=[0.1] * 8,
        event_summary="User plans a trip to Japan",
        event_summary_embedding=[0.1] * 8,
        metadata={"seed": True},
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
                                        "updates": [
                                            {
                                                "topic": "travel",
                                                "subtopic": "japan",
                                                "conv_id": "conv-2",
                                                "event_index": 0,
                                                "content": "Discuss hotels in Tokyo.",
                                                "action": "add",
                                                "metadata": {
                                                    "reason": "extend_event",
                                                    "is_continuation": True,
                                                    "event_summary_hint": "Trip planning with hotels",
                                                },
                                            }
                                        ],
                                        "recall_requests": [],
                                        "recent_topics": {
                                            "travel": [
                                                {
                                                    "subtopic": "japan",
                                                    "summary": "Trip planning with hotel research",
                                                }
                                            ]
                                        },
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
                                        "final_updates": [],
                                        "topic_merges": [],
                                        "context_updates": [
                                            "继续讨论东京酒店选择",
                                        ],
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
                                        "operations": [
                                            {
                                                "action": "add",
                                                "topic": "travel",
                                                "subtopic": "japan",
                                                "conv_id": "conv-2",
                                                "event_index": 0,
                                                "content": "Discuss hotels in Tokyo.",
                                                "metadata": {
                                                    "reason": "extend_event",
                                                    "is_continuation": True,
                                                },
                                            }
                                        ],
                                        "context_patch": "继续讨论东京酒店选择",
                                    }
                                )
                            }
                        }
                    ]
                }
            ]
        ),
    }

    def fake_generate(*, prompt: str, system_prompt: str):  # type: ignore[override]
        iterator = response_map.get(system_prompt)
        if iterator is None:
            raise AssertionError(f"Unexpected system prompt: {system_prompt}")
        return next(iterator)

    manager.llm_client.generate = fake_generate  # type: ignore[method-assign]

    agent = StructuredMemoryAgent(manager=manager)
    agent.ingest(conv_id="conv-2", role="user", content="What about hotels in Tokyo?")
    result = agent.run()

    stored_ids = result["prompt4"]["stored_memory_ids"]
    assert len(stored_ids) == 1

    topic_node = database.fetch_topic("travel", "japan")
    assert topic_node is not None
    assert "Tokyo" in topic_node.event_summary
    assert topic_node.event_summary_embedding is not None

    memories = database.fetch_memories(topic_node.id)
    assert len(memories) == 1
    assert memories[0]["metadata"]["is_continuation"] is True

    filtered = database.fetch_memories(topic_node.id, query_embedding=[0.1] * 8)
    assert filtered[0]["similarity"] == pytest.approx(1.0)

    assert manager.context["summary"] == "继续讨论东京酒店选择"
    assert manager.window_topics["travel"]["japan"]["summary"] == topic_node.event_summary


def test_topic_recall_branch_flow(tmp_path):
    manager, database = _make_manager(tmp_path)

    topic_id = database.upsert_topic(
        topic="travel",
        subtopic="paris",
        embedding=[0.1] * 8,
        event_summary="Explored Paris attractions",
        event_summary_embedding=[0.1] * 8,
        metadata={"seed": True},
    )
    database.add_memory(
        topic_id=topic_id,
        conv_id="past-1",
        event_index=0,
        content="Stayed near the Louvre.",
        embedding=[0.1] * 8,
        metadata={"action": "add", "source": "seed"},
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
                                        "updates": [],
                                        "recall_requests": [
                                            {
                                                "topic": "travel",
                                                "subtopic": "paris",
                                                "reason": "need_history",
                                            }
                                        ],
                                        "recent_topics": {},
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
                                        "tool_tasks": [
                                            {
                                                "tool": "topic_recall",
                                                "topic": "travel",
                                                "subtopic": "paris",
                                                "reason": "need_history",
                                            }
                                        ],
                                        "context_requirements": [],
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
                                        "final_updates": [
                                            {
                                                "topic": "travel",
                                                "subtopic": "paris",
                                                "conv_id": "conv-3",
                                                "event_index": 0,
                                                "content": "Revisiting the Paris trip details.",
                                                "action": "add",
                                                "metadata": {
                                                    "reason": "recall_and_extend",
                                                },
                                            }
                                        ],
                                        "topic_merges": [],
                                        "context_updates": ["巴黎旅行记忆补充"],
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
                                        "operations": [
                                            {
                                                "action": "add",
                                                "topic": "travel",
                                                "subtopic": "paris",
                                                "conv_id": "conv-3",
                                                "event_index": 0,
                                                "content": "Revisiting the Paris trip details.",
                                                "metadata": {
                                                    "reason": "recall_and_extend",
                                                },
                                            }
                                        ],
                                        "context_patch": "巴黎旅行记忆补充",
                                    }
                                )
                            }
                        }
                    ]
                }
            ]
        ),
    }

    def fake_generate(*, prompt: str, system_prompt: str):  # type: ignore[override]
        iterator = response_map.get(system_prompt)
        if iterator is None:
            raise AssertionError(f"Unexpected system prompt: {system_prompt}")
        return next(iterator)

    manager.llm_client.generate = fake_generate  # type: ignore[method-assign]

    agent = StructuredMemoryAgent(manager=manager)
    agent.ingest(conv_id="conv-3", role="user", content="Can you remind me about Paris?")
    result = agent.run()

    recall_results = result["prompt2"]["recall_results"]
    assert recall_results
    assert recall_results[0]["topic"] == "travel"
    assert recall_results[0]["memories"]
    assert recall_results[0]["memories"][0]["similarity"] == pytest.approx(1.0)

    stored_ids = result["prompt4"]["stored_memory_ids"]
    assert len(stored_ids) == 1

    topic_node = database.fetch_topic("travel", "paris")
    assert topic_node is not None
    memories = database.fetch_memories(topic_node.id)
    assert len(memories) == 2
    assert memories[-1]["content"].startswith("Revisiting the Paris trip")

    assert manager.context["summary"] == "巴黎旅行记忆补充"


def test_multi_event_window_processing(tmp_path):
    manager, database = _make_manager(tmp_path)

    agent = StructuredMemoryAgent(manager=manager)
    agent.ingest(conv_id="conv-4", role="user", content="Discuss project Alpha timeline.")
    agent.ingest(conv_id="conv-4", role="assistant", content="I hit the gym to keep fit.")
    agent.ingest(conv_id="conv-4", role="user", content="Project Alpha needs additional budget.")
    agent.ingest(conv_id="conv-4", role="assistant", content="Also planning a trip to Spain.")

    updates = [
        {
            "topic": "project",
            "subtopic": "alpha",
            "conv_id": "conv-4",
            "event_index": 0,
            "content": "Discuss project Alpha timeline.",
            "action": "add",
            "metadata": {"reason": "new_topic"},
        },
        {
            "topic": "personal",
            "subtopic": "fitness",
            "conv_id": "conv-4",
            "event_index": 1,
            "content": "I hit the gym to keep fit.",
            "action": "add",
            "metadata": {"reason": "new_topic"},
        },
        {
            "topic": "project",
            "subtopic": "alpha",
            "conv_id": "conv-4",
            "event_index": 2,
            "content": "Project Alpha needs additional budget.",
            "action": "add",
            "metadata": {"reason": "extend_event"},
        },
        {
            "topic": "travel",
            "subtopic": "spain",
            "conv_id": "conv-4",
            "event_index": 3,
            "content": "Also planning a trip to Spain.",
            "action": "add",
            "metadata": {"reason": "new_topic"},
        },
    ]

    response_map = {
        SYSTEM_PROMPT_1: iter(
            [
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "updates": updates,
                                        "recall_requests": [],
                                        "recent_topics": {
                                            "project": [
                                                {
                                                    "subtopic": "alpha",
                                                    "summary": "Project Alpha planning",
                                                }
                                            ],
                                            "personal": [
                                                {
                                                    "subtopic": "fitness",
                                                    "summary": "Fitness routine",
                                                }
                                            ],
                                            "travel": [
                                                {
                                                    "subtopic": "spain",
                                                    "summary": "Spain vacation ideas",
                                                }
                                            ],
                                        },
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
                                        "final_updates": updates,
                                        "topic_merges": [],
                                        "context_updates": [
                                            "项目与个人进展更新",
                                            "开始规划西班牙旅行",
                                        ],
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
                                        "operations": updates,
                                        "context_patch": "项目与个人进展更新\n开始规划西班牙旅行",
                                    }
                                )
                            }
                        }
                    ]
                }
            ]
        ),
    }

    def fake_generate(*, prompt: str, system_prompt: str):  # type: ignore[override]
        iterator = response_map.get(system_prompt)
        if iterator is None:
            raise AssertionError(f"Unexpected system prompt: {system_prompt}")
        return next(iterator)

    manager.llm_client.generate = fake_generate  # type: ignore[method-assign]

    result = agent.run()

    stored_ids = result["prompt4"]["stored_memory_ids"]
    assert len(stored_ids) == 4

    project_node = database.fetch_topic("project", "alpha")
    assert project_node is not None
    assert "timeline" in project_node.event_summary
    assert "budget" in project_node.event_summary

    personal_node = database.fetch_topic("personal", "fitness")
    assert personal_node is not None
    assert "keep fit" in personal_node.event_summary

    travel_node = database.fetch_topic("travel", "spain")
    assert travel_node is not None
    assert "Spain" in travel_node.event_summary

    project_memories = database.fetch_memories(project_node.id)
    assert len(project_memories) == 2
    personal_memories = database.fetch_memories(personal_node.id)
    assert len(personal_memories) == 1
    travel_memories = database.fetch_memories(travel_node.id)
    assert len(travel_memories) == 1

    assert manager.context["summary"] == "项目与个人进展更新\n开始规划西班牙旅行"
    assert manager.window_topics["project"]["alpha"]["summary"] == project_node.event_summary
    assert manager.window_topics["personal"]["fitness"]["summary"] == personal_node.event_summary
    assert manager.window_topics["travel"]["spain"]["summary"] == travel_node.event_summary
