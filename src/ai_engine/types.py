# Copyright (c) 2026 Frank1o3
# SPDX-License-Identifier: MIT

"""Core type definitions for the GropWave AI Engine in Python 3.14.
Defines model tiers, quota structures,
task classifications, messages, and configurations.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal


class TaskTier(StrEnum):
    """Expertise tier for model routing."""

    FAST = "fast"  # 8B, mini models — indexing, summaries, simple Qs
    BALANCED = "balanced"  # 13B-34B — refactoring, explanations
    HEAVY = "heavy"  # 70B+, pro — complex logic, debugging, code gen


class QuotaHealth(StrEnum):
    """Health status of a model's quota."""

    HEALTHY = "healthy"  # < warning threshold
    WARNING = "warning"  # >= warning threshold but < 100%
    EXHAUSTED = "exhausted"  # >= 100% — should not be used


# Hard-coded tier assignment by model ID pattern
MODEL_TIER_MAP: dict[str, TaskTier] = {
    # Fast tier — lightweight models with high daily limits
    "llama-3.1-8b-instant": TaskTier.FAST,
    "llama3-8b-8192": TaskTier.FAST,
    "gemma2-9b-it": TaskTier.FAST,
    "llama-3.2-11b-vision-preview": TaskTier.FAST,
    "llama-3.2-90b-vision-preview": TaskTier.BALANCED,
    # Balanced tier
    "llama-3.1-70b-versatile": TaskTier.BALANCED,
    "llama3-70b-8192": TaskTier.BALANCED,
    "mixtral-8x7b-32768": TaskTier.BALANCED,
    "llama-guard-4-12b": TaskTier.BALANCED,
    # Heavy tier — pro / largest models
    "llama-3.1-405b-reasoning": TaskTier.HEAVY,
}


def resolve_tier(model_id: str) -> TaskTier:
    """Resolve task tier from a model ID string using direct map or pattern heuristics."""
    if model_id in MODEL_TIER_MAP:
        return MODEL_TIER_MAP[model_id]

    lower = model_id.lower()
    if any(k in lower for k in ("8b", "9b", "11b", "mini", "instant", "allam")):
        return TaskTier.FAST
    if any(k in lower for k in ("405b", "pro", "reasoning", "scout")):
        return TaskTier.HEAVY
    return TaskTier.BALANCED


@dataclass(slots=True)
class ModelLimits:
    """Quota limits for a single model."""

    rpm: float = 30.0  # requests per minute
    rpd: float = 1000.0  # requests per day
    tpm: float = 6000.0  # tokens per minute
    tpd: float = 500000.0  # tokens per day


@dataclass(slots=True)
class QuotaUsage:
    """Current quota usage snapshot for a model."""

    rpm_count: int = 0
    rpd_count: int = 0
    tpm_count: int = 0
    tpd_count: int = 0


@dataclass(slots=True)
class ConstrainedDimension:
    """Represents the most constrained limit dimension."""

    key: str
    ratio: float


@dataclass(slots=True)
class ModelQuotaStatus:
    """Combined status of a model including quota health."""

    model_id: str
    health: QuotaHealth
    limits: ModelLimits
    usage: QuotaUsage
    most_constrained: ConstrainedDimension


@dataclass(slots=True)
class RegisteredModel:
    """A model entry in the registry."""

    id: str
    name: str
    tier: TaskTier
    limits: ModelLimits
    disabled: bool = False


@dataclass(slots=True)
class TaskMeta:
    """Metadata regarding task signals."""

    file_count: int = 0
    has_terminal_output: bool = False
    has_code_block: bool = False
    prompt_length: int = 0


@dataclass(slots=True)
class ClassifiedTask:
    """A categorized task ready for model selection."""

    prompt: str
    tier: TaskTier
    estimated_tokens: int
    meta: TaskMeta


TASK_CLASSIFY_THRESHOLD = {
    "fast_max_tokens": 2000,
    "fast_max_files": 5,
    "heavy_min_tokens": 4000,
    "heavy_has_code": True,
}


@dataclass(slots=True)
class OrchestratorConfig:
    """Configuration options for OrchestratorCore."""

    api_key: str = ""
    base_url: str = ""
    default_model: str = "auto"
    quota_warning_threshold: float = 0.9
    cache_directory: str = ".groqwave_cache"
    context_file: str = "context.md"
    system_file: str = "system.md"
    memory_file: str = "memory.md"


@dataclass(slots=True)
class ChatMessage:
    """Represents a chat message."""

    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: float = 0.0
    model_id: str | None = None


@dataclass(slots=True)
class ParsedToolCall:
    """Parsed tool call representation."""

    type: Literal["run_command", "edit_file"]
    payload: str
    target_path: str | None = None


@dataclass(slots=True)
class TerminalResult:
    """Result of a command execution."""

    command: str
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool = False


@dataclass(slots=True)
class FileEditResult:
    """Result of a file editing operation."""

    success: bool
    message: str


@dataclass(slots=True)
class MemoryEntry:
    """Memory entry model for persistent memory."""

    id: str
    type: Literal["code_summary", "conversation", "observation"]
    source: str
    created: str
    last_accessed: str
    access_count: int
    tags: list[str] = field(default_factory=list)
    decay_score: float = 1.0
    content: str = ""
