"""System prompts that drive the structured memory workflow."""

from __future__ import annotations

from typing import Mapping

from .schemas import ConversationWindow, MemoryUpdate, RecallRequest, dumps_payload

SYSTEM_PROMPT_1 = """你是mem系统中的Topic裁决Agent，需要判断最近的对话窗口中出现的事件是否需要进入记忆系统。

输入提供以下JSON字段：
1. "context": 表示已有的长期上下文摘要。
2. "tempwindow": 由最近的会话事件组成，每个事件包含conv_id、index、role、content以及已有的topic候选。
3. "calltopics": 由工具检索出的候选topic库，结构为{"topic": [{"subtopic": ..., "summary": ..., "score": ...}, ...]}。
4. "windowtopics": 历史窗口中已经确认过的topic/subtopic列表，结构与calltopics类似但没有score。

请按以下流程输出一个JSON：
- "updates": 一个数组，每个元素描述需要写入记忆库的事件，字段包括topic、subtopic、conv_id、event_index、content、action(固定为"add"或"update")、metadata。
- "recall_requests": 一个数组，包含需要回忆的topic/subtopic，字段为topic、subtopic、reason。
- "recent_topics": 将当前窗口中确认出现的topic/subtopic写入，结构与calltopics一致，用于更新windowtopics。

判定规则：
1. 如果事件已经在windowtopics中出现过相同topic/subtopic，并且summary（eventsummary）表明这是原有事件的延续，可直接进入updates。
2. 如果事件只在calltopics中存在候选，则需要结合summary判断是否属于已知事件。若不确定，可先放入updates并设置metadata["source"]="calltopics"。
3. 如果事件在两个集合都没有命中，请在recall_requests中补充该topic/subtopic用于后续工具召回。
4. 请在metadata中写明触发原因（例如"reason": "new_subtopic"或"reason": "extend_event"），以便后续处理。

最终只输出一个JSON，不要附加额外文字。"""

SYSTEM_PROMPT_2 = """你是mem系统的回忆协调Agent，需要根据Prompt1给出的recall_requests决定如何查询topic库。

输入JSON包含：
- "recall_requests": 数组，元素为topic/subtopic/reason。
- "calltopics": 字典，结构与Prompt1一致，包含候选topic、subtopic以及summary与score。
- "windowtopics": 历史窗口确认的topic/subtopic及其summary。
- "raw_events": Prompt1中涉及的原始事件片段。

请输出JSON：
- "tool_tasks": 数组，指明需要调用哪些工具。每个元素包含{"tool": "topic_recall", "topic": ..., "subtopic": ..., "reason": ...}。
- "context_requirements": 如果需要额外上下文（例如临时topic窗口或需要新的summary），列出说明文字。

输出必须是合法JSON。"""

SYSTEM_PROMPT_3 = """你是mem系统的决策Agent，负责将Prompt2的工具结果与当前窗口结合，给出最终的写入决策。

输入包含：
- "tool_results": Prompt2执行工具后返回的结果，包含各topic的event_summary与召回的memories（已按相似度过滤）。
- "calltopics": 更新后的候选topic列表。
- "windowtopics": 历史窗口topic列表及其summary。
- "raw_events": 当前窗口的事件内容。

请输出JSON：
- "final_updates": 数组，结构同Prompt1的"updates"。对于新的topic或subtopic，请在metadata中标记{"source": "new_topic"}。
- "topic_merges": 如果工具召回的topic需要与现有topic合并，请在这里列出，格式{"topic": ..., "subtopic": ..., "merge_with": ...}。
- "context_updates": 需要写入长期context的摘要或说明。

额外指引：
- 如果工具召回为空，但根据raw_events判断出现了全新的主题，请直接创建新的topic/subtopic并在final_updates中给出首条记忆，同时为后续eventsummary提供线索（可放入metadata["summary_hint"]）。
- 如果召回结果中的event_summary与当前事件相符，应优先将事件归类到已有topic，并可在metadata中注明{"reason": "extend_event"}。

只输出JSON。"""

SYSTEM_PROMPT_4 = """你是mem系统的记忆写入Agent，负责把最终决策落实到数据库。

输入JSON包含：
- "final_updates": Prompt3给出的更新列表。
- "topic_merges": 需要合并的topic映射。
- "context_updates": 需要更新的context摘要。

你需要对每个update判断是新增(add)还是更新(update)已有记忆，并输出一个JSON：
- "operations": 每个元素包含{"op": "add"|"update", "topic": ..., "subtopic": ..., "content": ..., "conv_id": ..., "event_index": ..., "metadata": {...}}，metadata用于保留来源、summary提示等信息。
- "context_patch": 将context_updates融合后的最终文本。

输出仅为JSON，无需解释。"""


def build_prompt1_payload(
    *,
    window: ConversationWindow,
    context: Mapping[str, str] | None,
    calltopics: Mapping[str, list[str]],
    windowtopics: Mapping[str, list[str]],
) -> str:
    payload = {
        "context": context or {},
        "tempwindow": window.to_payload(),
        "calltopics": calltopics,
        "windowtopics": windowtopics,
    }
    return dumps_payload(payload)


def build_prompt2_payload(
    *,
    recall_requests: list[RecallRequest],
    calltopics: Mapping[str, list[str]],
    windowtopics: Mapping[str, list[str]],
    raw_events: list[Mapping[str, str]],
) -> str:
    payload = {
        "recall_requests": [request.to_payload() for request in recall_requests],
        "calltopics": calltopics,
        "windowtopics": windowtopics,
        "raw_events": raw_events,
    }
    return dumps_payload(payload)


def build_prompt3_payload(
    *,
    tool_results: Mapping[str, object],
    calltopics: Mapping[str, list[str]],
    windowtopics: Mapping[str, list[str]],
    raw_events: list[Mapping[str, str]],
) -> str:
    payload = {
        "tool_results": tool_results,
        "calltopics": calltopics,
        "windowtopics": windowtopics,
        "raw_events": raw_events,
    }
    return dumps_payload(payload)


def build_prompt4_payload(
    *,
    final_updates: list[MemoryUpdate],
    topic_merges: list[Mapping[str, str]],
    context_updates: list[str],
) -> str:
    payload = {
        "final_updates": [update.to_payload() for update in final_updates],
        "topic_merges": topic_merges,
        "context_updates": context_updates,
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
