# Copyright (c) 2026 Frank1o3
# SPDX-License-Identifier: MIT

"""TaskRouter — classifies user prompts into task tiers using lightweight heuristics.

Classification heuristics:
  - Token estimation
  - Presence of code blocks
  - Presence of terminal output / tracebacks / error messages
  - Number of active workspace files in context
  - Keyword patterns (index, summarize vs debug, refactor, implement)
"""

import math
import re

from .types import (
    TASK_CLASSIFY_THRESHOLD,
    ClassifiedTask,
    TaskMeta,
    TaskTier,
)

FAST_KEYWORDS = [
    "index",
    "summarize",
    "summary",
    "list",
    "what does",
    "explain briefly",
    "quick",
    "simple",
    "outline",
    "overview",
    "describe",
]

HEAVY_KEYWORDS = [
    "debug",
    "fix",
    "refactor",
    "architecture",
    "implement",
    "write",
    "generate",
    "optimize",
    "complex",
    "bug",
    "error",
    "stack trace",
    "why is",
    "how do i",
    "build",
    "create a function",
    "create a class",
]


def estimate_tokens(text: str) -> int:
    """Rough approximation: ~1.3 tokens per word for English code-mixed text."""
    words = [w for w in text.split() if w]
    return math.ceil(len(words) * 1.3)


def has_code_block(text: str) -> bool:
    """Detect fenced code blocks or common programming signatures."""
    if re.search(r"```[\s\S]*```", text):
        return True
    return bool(
        re.search(
            r"^(import |export |function |class |const |let |def |async def )",
            text,
            re.MULTILINE,
        )
    )


def has_terminal_output(text: str) -> bool:
    """Detect terminal output or error tracebacks."""
    return bool(
        re.search(
            r"(error|traceback|exception|segmentation fault|command not found|ENOENT|EACCES|ReferenceError|TypeError|ValueError)",
            text,
            re.IGNORECASE,
        )
    )


def contains_keywords(text: str, keywords: list[str]) -> bool:
    """Check if lowercased text contains any keyword from the list."""
    lower = text.lower()
    return any(kw in lower for kw in keywords)


class TaskRouter:
    """Fast prompt classifier for model routing."""

    def classify(self, prompt: str, file_count: int = 0) -> ClassifiedTask:
        """Classify a prompt into a TaskTier with metadata."""
        tokens = estimate_tokens(prompt)
        code = has_code_block(prompt)
        terminal = has_terminal_output(prompt)
        prompt_len = len(prompt)

        meta = TaskMeta(
            file_count=file_count,
            has_terminal_output=terminal,
            has_code_block=code,
            prompt_length=prompt_len,
        )

        tier = self._decide_tier(prompt, tokens, code, terminal, file_count)

        return ClassifiedTask(
            prompt=prompt,
            tier=tier,
            estimated_tokens=tokens,
            meta=meta,
        )

    def _decide_tier(
        self,
        prompt: str,
        tokens: int,
        has_code: bool,
        has_terminal: bool,
        file_count: int,
    ) -> TaskTier:
        fast_max_tokens = TASK_CLASSIFY_THRESHOLD["fast_max_tokens"]
        fast_max_files = TASK_CLASSIFY_THRESHOLD["fast_max_files"]
        heavy_min_tokens = TASK_CLASSIFY_THRESHOLD["heavy_min_tokens"]

        # -- Heavy signals --
        if tokens >= heavy_min_tokens:
            return TaskTier.HEAVY
        if contains_keywords(prompt, HEAVY_KEYWORDS) and (has_code or has_terminal):
            return TaskTier.HEAVY
        if has_terminal and has_code:
            return TaskTier.HEAVY

        # -- Fast signals --
        if tokens <= fast_max_tokens and file_count <= fast_max_files:
            if contains_keywords(prompt, FAST_KEYWORDS):
                return TaskTier.FAST
            if not has_code and tokens < 500:
                return TaskTier.FAST

        # -- Default to balanced --
        return TaskTier.BALANCED
