"""Structured memory system aligned with topic-based architecture.

This subpackage exposes a high-level manager that implements the six-step
architecture described in the design brief.  It wires together

* prompt templates for the four system prompts,
* local LLM / embedding clients that can call a locally hosted model,
* persistent storage for topic / subtopic memories with vector embeddings, and
* tools that the agent can use to retrieve, update, and summarise memories.
"""

from .clients import LLMClient
from .manager import StructuredMemoryAgent, StructuredMemoryManager
from .runtime import StructuredMemoryRuntime, main as runtime_main
from .prompts import (
    SYSTEM_PROMPT_1,
    SYSTEM_PROMPT_2,
    SYSTEM_PROMPT_3,
    SYSTEM_PROMPT_4,
    TOPIC_EVENT0_PROMPT,
)
from .schemas import (
    ConversationEvent,
    ConversationWindow,
    MemoryUpdate,
    TopicCandidate,
    TopicNode,
)

__all__ = [
    "ConversationEvent",
    "ConversationWindow",
    "LLMClient",
    "MemoryUpdate",
    "StructuredMemoryAgent",
    "StructuredMemoryManager",
    "StructuredMemoryRuntime",
    "TOPIC_EVENT0_PROMPT",
    "TopicCandidate",
    "TopicNode",
    "SYSTEM_PROMPT_1",
    "SYSTEM_PROMPT_2",
    "SYSTEM_PROMPT_3",
    "SYSTEM_PROMPT_4",
    "runtime_main",
]
