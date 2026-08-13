# Copyright (c) 2026 Frank1o3
# SPDX-License-Identifier: MIT

"""MemoryEngine — unified scored memory store in Python.

Manages memory.md in the workspace directory.
Supports code summaries, conversation turns,
and observations with score decay and relevance retrieval.
"""

import math
import random
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from .system_prompt_engine import extract_keywords
from .types import MemoryEntry


def short_uuid() -> str:
    """Generate an 8-character hex string ID."""
    return f"{random.randint(0, 0xFFFFFFFF):08x}"


def serialize_entry(entry: MemoryEntry) -> str:
    """Serialize MemoryEntry into a YAML frontmatter block."""
    tags_str = f"[{', '.join(f'"{t}"' for t in entry.tags)}]" if entry.tags else "[]"
    return "\n".join([
        "---",
        f"id: {entry.id}",
        f"type: {entry.type}",
        f"source: {entry.source}",
        f"created: {entry.created}",
        f"last_accessed: {entry.last_accessed}",
        f"access_count: {entry.access_count}",
        f"decay_score: {entry.decay_score:.4f}",
        f"tags: {tags_str}",
        "---",
        entry.content,
        "\n",
    ])


def parse_entries(content: str) -> list[MemoryEntry]:
    """Parse all MemoryEntry blocks from memory.md."""
    entries: list[MemoryEntry] = []
    pattern = re.compile(
        r"---\nid:\s*(.+?)\ntype:\s*(.+?)\nsource:\s*(.+?)\ncreated:\s*(.+?)\nlast_accessed:\s*(.+?)\naccess_count:\s*(\d+)\ndecay_score:\s*([\d.]+)\ntags:\s*(\[.*?\])\n---\n([\s\S]*?)(?=\n---\nid:|$)"
    )

    for match in pattern.finditer(content):
        raw_tags = match.group(8).strip()
        tags: list[str] = []
        if len(raw_tags) > 2:
            inner = raw_tags[1:-1].strip()
            if inner:
                tags = [t.strip().strip("\"'") for t in inner.split(",") if t.strip()]

        entries.append(
            MemoryEntry(
                id=match.group(1).strip(),
                type=match.group(2).strip(),  # type: ignore[arg-type]
                source=match.group(3).strip(),
                created=match.group(4).strip(),
                last_accessed=match.group(5).strip(),
                access_count=int(match.group(6)),
                decay_score=float(match.group(7)),
                tags=tags,
                content=match.group(9).strip(),
            )
        )

    return entries


class MemoryEngine:
    """Memory store performing frontmatter parsing, score decay, and relevance retrieval."""

    def __init__(
        self,
        memory_filename: str = "memory.md",
        workspace_root: Path | str | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root) if workspace_root else Path.cwd()
        self.memory_file_path = self.workspace_root / memory_filename
        self._entries: list[MemoryEntry] = []
        self._initialized = False

    def load(self) -> None:
        """Load existing entries from memory.md."""
        if not self.memory_file_path.exists():
            self._entries = []
        else:
            try:
                content = self.memory_file_path.read_text(encoding="utf-8")
                self._entries = parse_entries(content)
            except Exception:
                self._entries = []
        self._initialized = True

    def decay_all(self) -> None:
        """Decay all entries scores by multiplying by 0.92."""
        for entry in self._entries:
            entry.decay_score *= 0.92
        self._persist_entries()

    def add_entry(
        self,
        type_: Literal["code_summary", "conversation", "observation"],
        source: str,
        content: str,
        tags: list[str],
    ) -> MemoryEntry:
        """Add a new memory entry and persist to memory.md."""
        now = datetime.now(UTC).isoformat()
        entry = MemoryEntry(
            id=short_uuid(),
            type=type_,
            source=source,
            created=now,
            last_accessed=now,
            access_count=0,
            tags=tags,
            decay_score=1.0,
            content=content,
        )

        self._entries.append(entry)
        self._persist_entries()
        return entry

    def retrieve_relevant(self, prompt: str, max_entries: int = 8) -> str:
        """Retrieve entries relevant to user's prompt scored by overlap, recency, and decay."""
        prompt_keywords = extract_keywords(prompt, 30)
        if not prompt_keywords:
            return ""

        now_ts = datetime.now(UTC).timestamp()
        one_day_sec = 86400.0

        scored: list[tuple[MemoryEntry, float]] = []

        for entry in self._entries:
            if entry.decay_score < 0.1:
                continue

            overlap_score = 0.0
            for kw in prompt_keywords:
                if kw in entry.tags:
                    overlap_score += 1.0

            content_lower = entry.content.lower()
            for kw in prompt_keywords:
                if kw in content_lower:
                    overlap_score += 0.3

            try:
                dt = datetime.fromisoformat(entry.last_accessed)
                last_accessed_ts = dt.timestamp()
            except Exception:
                last_accessed_ts = now_ts

            days_since = max(0.0, (now_ts - last_accessed_ts) / one_day_sec)
            recency_boost = math.exp(-days_since / 30.0)
            access_boost = min(entry.access_count * 0.05, 0.5)

            total_score = (
                overlap_score + recency_boost + access_boost
            ) * entry.decay_score
            scored.append((entry, total_score))

        scored.sort(key=lambda item: item[1], reverse=True)
        top = scored[:max_entries]

        if not top:
            return ""

        results: list[str] = []
        for entry, _ in top:
            header = f"[{entry.type}] {entry.source}"
            results.append(f"### {header}\n{entry.content}")

        return "\n\n".join(results)

    def record_access(self, entry_id: str) -> None:
        """Record entry access by updating last_accessed timestamp and access_count."""
        for entry in self._entries:
            if entry.id == entry_id:
                entry.last_accessed = datetime.now(UTC).isoformat()
                entry.access_count += 1
                self._persist_entries()
                break

    def add_conversation_turn(
        self, user_prompt: str, assistant_response: str, model_id: str
    ) -> None:
        """Add a conversation turn as a memory entry."""
        combined = f"User: {user_prompt}\nAssistant: {assistant_response}"
        tags = extract_keywords(combined, 12)
        truncated = combined[:800] + "..." if len(combined) > 800 else combined
        source = f"chat ({model_id})" if model_id else "chat"
        self.add_entry("conversation", source, truncated, tags)

    def prune_decayed(self, threshold: float = 0.05) -> int:
        """Remove entries with decay_score below threshold."""
        before = len(self._entries)
        self._entries = [e for e in self._entries if e.decay_score >= threshold]
        removed = before - len(self._entries)
        if removed > 0:
            self._persist_entries()
        return removed

    def get_all_entries(self) -> list[MemoryEntry]:
        """Return all memory entries."""
        return list(self._entries)

    def _persist_entries(self) -> None:
        header = (
            f"# GropWave Memory\n"
            f"# Generated: {datetime.now(UTC).isoformat()}\n"
            f"# Entries: {len(self._entries)}\n\n"
        )
        content = header + "\n".join(serialize_entry(e) for e in self._entries)
        self.memory_file_path.write_text(content, encoding="utf-8")
