"""系统提示词（中文）用于指导结构化记忆流程。"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from .schemas import ConversationEvent, dumps_payload

SYSTEM_PROMPT_1 = """你是 mem 结构化记忆系统中的「事件判定 Agent」。

【任务目标】
1. 判断当前对话事件（event）是否属于已存在的 topic/subtopic 下的事件延续。
2. 如不是延续，判断是否需要触发 topic/subtopic 的召回或新建。
3. 为后续事件总结提供摘要关注点。

【核心概念】
- event[0]：单个事件的最小记忆单元。若判定为已有事件的延续，需要依据已有的 event_summary 更新。
- 事件延续：当事件内容与某个 topic/subtopic 的标签与 event_summary 一致地描述同一持续话题、同一场景或连续行动时视为延续。
  例：已存在 topic="旅行"、subtopic="日本行程"，event_summary=“计划日本行程并安排航班”。
      当前事件“我们开始预订东京酒店吗？” => 延续。
      当前事件“要不要改成去韩国？” => 不是延续，需要新分支。
- tempWindow：一次处理的会议/对话窗口。我们逐条处理窗口内的对话，每条对话只归属到一个事件。

【输入字段】
- "context_summary"：当前长期上下文摘要，可为空字符串。
- "tempwindow"：包含
  * "window_id"
  * "position"：本事件在本 tempWindow 中的序号（从 1 开始）。
  * "event"：当前事件，字段为 conv_id/index/role/content/metadata。
  * "history"：同一 tempWindow 中，紧邻当前事件之前的若干事件内容（按时间排序）。
- "calltopics"：向量检索得到的候选 topic/subtopic 信息。
- "windowtopics"：本 tempWindow 已确认的 topic/subtopic 及其 event_summary。
- "topic_summaries"：数据库中保存的 topic/subtopic 与最新 event_summary。

【输出 JSON】
{
  "decision": {
    "topic": "...",              // 判定的 topic 名称，如果无法确定使用空字符串
    "subtopic": "...",           // 判定的 subtopic 名称
    "status": "continuation" | "new_subtopic" | "new_topic",
    "need_recall": true | false,   // 是否需要额外召回
    "create_memory": true | false, // 是否需要为该事件写入记忆
    "reason": "..."               // 判定理由，解释延续或新建的原因
  },
  "summary_clues": ["...", "..."] // 可选，列出 1~3 个提示，说明总结时应关注的细节
}

请仅返回 JSON，不要包含额外文本。
"""

SYSTEM_PROMPT_2 = """你是 mem 系统的「召回规划 Agent」。

【任务目标】
基于 Prompt1 的决策，如果需要召回（decision.need_recall=true），给出需要查询的 topic/subtopic 以及检索线索。

【输入字段】
- "decision"：Prompt1 的 decision 字段。
- "event"：当前事件（conv_id/index/role/content/metadata）。
- "calltopics"、"windowtopics"：与 Prompt1 相同。

【输出 JSON】
{
  "recall_tasks": [
    {
      "topic": "...",          // 待召回的 topic
      "subtopic": "...",       // 待召回的 subtopic，可为空字符串表示只知道大 topic
      "query": "..."           // 召回时使用的查询描述，突出与当前事件相关的要点
    }
  ],
  "notes": "可选的补充说明"
}

若无需召回，recall_tasks 返回空数组。
只输出 JSON。
"""

SYSTEM_PROMPT_3 = """你是 mem 系统的「决策整合 Agent」。

【任务目标】
结合 Prompt1 的决策、工具召回结果以及当前事件，确认最终 topic/subtopic，并整理事件总结要点。

【输入字段】
- "decision"：Prompt1 的 decision。
- "event"：当前事件内容。
- "recall_results": 工具返回的召回结果数组，每项包含 topic/subtopic/event_summary/memories。
- "calltopics"、"windowtopics"：参考信息。

【输出 JSON】
{
  "final_topic": {
    "topic": "...",
    "subtopic": "...",
    "status": "continuation" | "new_subtopic" | "new_topic",
    "create_memory": true | false,
    "reason": "..." // 若与 Prompt1 不同，请解释原因
  },
  "event_summary_focus": "...", // 对 event_summary 更新时应关注的重点
  "memory_outline": ["...", "..."], // 记忆内容的关键点（1~4 条）
  "context_note": "..." // 可选，对长期 context 的补充描述
}

若最终无需写入记忆，可将 final_topic.create_memory 设为 false，并说明原因。
只输出 JSON。
"""

SYSTEM_PROMPT_4 = """你是 mem 系统的「写入执行 Agent」。

【任务目标】
根据最终决策，为数据库写入准备结构化结果，包括：
1. 生成 event[0] 的最新摘要（event_summary）。
2. 生成需要存储的记忆文本。
3. 给出更新 context 的文本。

【输入字段】
- "final_topic"：Prompt3 的结果。
- "event"：当前事件。
- "history"：当前事件之前的简要对话列表。
- "previous_summary"：该 topic/subtopic 现有的 event_summary，可为空。
- "summary_focus"：Prompt3 提供的关注点。
- "memory_outline"：Prompt3 给出的要点列表。
- "context_note"：Prompt3 的 context_note。

【输出 JSON】
{
  "memory_text": "...",   // 建议写入的记忆文本，若 create_memory=false 可为空字符串
  "event_summary": "...", // 更新后的 event_summary
  "context_patch": "...", // 更新后的上下文摘要，可为空
  "metadata": {"tags": ["..."], "reason": "..."} // 可选，补充信息
}

请确保 event_summary 为自然语言摘要，可用于判断后续事件是否延续。
只输出 JSON。
"""


def build_prompt1_payload(
    *,
    context_summary: str | None,
    window_id: str,
    position: int,
    event: ConversationEvent,
    history: Sequence[ConversationEvent],
    calltopics: Mapping[str, list[Mapping[str, object]]],
    windowtopics: Mapping[str, list[Mapping[str, object]]],
    topic_summaries: Iterable[Mapping[str, str]],
) -> str:
    payload = {
        "context_summary": context_summary or "",
        "tempwindow": {
            "window_id": window_id,
            "position": position,
            "event": event.to_payload(),
            "history": [item.to_payload() for item in history],
        },
        "calltopics": calltopics,
        "windowtopics": windowtopics,
        "topic_summaries": list(topic_summaries),
    }
    return dumps_payload(payload)


def build_prompt2_payload(
    *,
    decision: Mapping[str, object],
    event: ConversationEvent,
    calltopics: Mapping[str, list[Mapping[str, object]]],
    windowtopics: Mapping[str, list[Mapping[str, object]]],
) -> str:
    payload = {
        "decision": decision,
        "event": event.to_payload(),
        "calltopics": calltopics,
        "windowtopics": windowtopics,
    }
    return dumps_payload(payload)


def build_prompt3_payload(
    *,
    decision: Mapping[str, object],
    event: ConversationEvent,
    recall_results: Sequence[Mapping[str, object]],
    calltopics: Mapping[str, list[Mapping[str, object]]],
    windowtopics: Mapping[str, list[Mapping[str, object]]],
) -> str:
    payload = {
        "decision": decision,
        "event": event.to_payload(),
        "recall_results": list(recall_results),
        "calltopics": calltopics,
        "windowtopics": windowtopics,
    }
    return dumps_payload(payload)


def build_prompt4_payload(
    *,
    final_topic: Mapping[str, object],
    event: ConversationEvent,
    history: Sequence[Mapping[str, object]],
    previous_summary: str | None,
    summary_focus: str | None,
    memory_outline: Sequence[str] | None,
    context_note: str | None,
) -> str:
    payload = {
        "final_topic": final_topic,
        "event": event.to_payload(),
        "history": list(history),
        "previous_summary": previous_summary or "",
        "summary_focus": summary_focus or "",
        "memory_outline": list(memory_outline or []),
        "context_note": context_note or "",
    }
    return dumps_payload(payload)


__all__ = [
    "SYSTEM_PROMPT_1",
    "SYSTEM_PROMPT_2",
    "SYSTEM_PROMPT_3",
    "SYSTEM_PROMPT_4",
    "build_prompt1_payload",
    "build_prompt2_payload",
    "build_prompt3_payload",
    "build_prompt4_payload",
]
