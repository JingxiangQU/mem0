"""Structured memory system aligned with topic-based architecture.

This subpackage exposes a high-level manager that implements the six-step
architecture described in the design brief.  It wires together

* prompt templates for the four system prompts,
* local LLM / embedding clients that can call a locally hosted model,
* persistent storage for topic / subtopic memories with vector embeddings, and
* tools that the agent can use to retrieve, update, and summarise memories.
"""

from .clients import LocalEmbeddingClient, LocalEmbeddingConfig, LocalLLMClient, LocalLLMConfig
from .manager import StructuredMemoryAgent, StructuredMemoryManager
from .runtime import StructuredMemoryRuntime, main as runtime_main
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
    TopicCandidate,
    TopicNode,
)

__all__ = [
    "ConversationEvent",
    "ConversationWindow",
    "LocalEmbeddingClient",
    "LocalEmbeddingConfig",
    "LocalLLMClient",
    "LocalLLMConfig",
    "MemoryUpdate",
    "RecallRequest",
    "StructuredMemoryAgent",
    "StructuredMemoryManager",
    "StructuredMemoryRuntime",
    "runtime_main",
    "SYSTEM_PROMPT_1",
    "SYSTEM_PROMPT_2",
    "SYSTEM_PROMPT_3",
    "SYSTEM_PROMPT_4",
    "TopicCandidate",
    "TopicNode",
    "build_prompt1_payload",
    "build_prompt2_payload",
    "build_prompt3_payload",
    "build_prompt4_payload",
]
