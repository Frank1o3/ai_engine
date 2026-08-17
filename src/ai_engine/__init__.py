"""GropWave AI Engine — Python 3.14 package.

Provides model orchestration, quota tracking, system prompt section extraction,
workspace context indexing, memory store decay/retrieval, active scope tracking,
and agentic tool execution.
"""

from .active_context import ActiveContext, ActiveFileTracker
from .agent_tools import (
    edit_file_content,
    execute_tool_calls_async,
    execute_tool_calls_sync,
    is_dangerous_command,
    parse_tool_calls,
    run_command_async,
    run_command_sync,
)
from .context_engine import ContextEngine, FileSummary, IndexedSymbol
from .groq_client import GroqClient
from .memory_engine import MemoryEngine
from .models import ModelRegistry
from .orchestrator import OrchestratorCore
from .quota import QuotaTracker
from .router import TaskRouter
from .system_prompt_engine import SystemPromptEngine
from .types import (
    ChatMessage,
    ClassifiedTask,
    FileEditResult,
    MemoryEntry,
    ModelLimits,
    ModelQuotaStatus,
    OrchestratorConfig,
    ParsedToolCall,
    QuotaHealth,
    QuotaUsage,
    RegisteredModel,
    TaskMeta,
    TaskTier,
    TerminalResult,
)

__all__ = [
    "ActiveContext",
    "ActiveFileTracker",
    "ChatMessage",
    "ClassifiedTask",
    "ContextEngine",
    "FileEditResult",
    "FileSummary",
    "GroqClient",
    "IndexedSymbol",
    "MemoryEngine",
    "MemoryEntry",
    "ModelLimits",
    "ModelQuotaStatus",
    "ModelRegistry",
    "OrchestratorConfig",
    "OrchestratorCore",
    "ParsedToolCall",
    "QuotaHealth",
    "QuotaTracker",
    "QuotaUsage",
    "RegisteredModel",
    "SystemPromptEngine",
    "TaskMeta",
    "TaskRouter",
    "TaskTier",
    "TerminalResult",
    "edit_file_content",
    "execute_tool_calls_async",
    "execute_tool_calls_sync",
    "is_dangerous_command",
    "parse_tool_calls",
    "run_command_async",
    "run_command_sync",
]
