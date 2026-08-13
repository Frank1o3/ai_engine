# Copyright (c) 2026 Frank1o3
# SPDX-License-Identifier: MIT

"""ContextEngine — manages workspace indexing and context.md generation.

Features:
  - Scans workspace files respecting .gitignore rules.
  - Summarizes files via lightweight LLM calls (with 429 exponential backoff retry).
  - Provides AST/regex fallback symbol & dependency extractors when LLM is offline.
  - Generates markdown structured documentation for context.md.
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from .groq_client import GroqClient
    from .types import OrchestratorConfig

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FileSummary:
    """Structured summary of a file."""

    relative_path: str
    purpose: str
    key_elements: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)


DEFAULT_EXCLUSIONS = [
    "**/node_modules/**",
    "**/.git/**",
    "**/.cache/**",
    "**/.venv/**",
    "**/__pycache__/**",
    "**/venv/**",
    "**/env/**",
    "**/dist/**",
    "**/out/**",
    "**/build/**",
    "**/*.lock",
    "**/package-lock.json",
    "**/poetry.lock",
    "**/Pipfile.lock",
]


class ContextEngine:
    """Manages reading, writing, indexing, and incremental updating of context.md."""

    def __init__(
        self,
        config: OrchestratorConfig,
        groq_client: GroqClient | None = None,
        workspace_root: Path | str | None = None,
    ) -> None:
        self.config = config
        self.groq_client = groq_client
        self.workspace_root = Path(workspace_root) if workspace_root else Path.cwd()

    def get_context_path(self) -> Path:
        """Get absolute path to context.md."""
        return self.workspace_root / self.config.context_file

    def read_context(self) -> str:
        """Read context.md contents. Returns empty string if missing."""
        ctx_path = self.get_context_path()
        if not ctx_path.exists():
            return ""
        try:
            return ctx_path.read_text(encoding="utf-8")
        except Exception:
            return ""

    def write_context(self, content: str) -> None:
        """Write content to context.md."""
        ctx_path = self.get_context_path()
        ctx_path.write_text(content, encoding="utf-8")

    # ─── Indexing ──────────────────────────────────────────────────────────────

    def index_workspace(
        self, progress_callback: Callable[[int, int], None] | None = None
    ) -> str:
        """Synchronously scan workspace files and generate context.md."""
        files = self._scan_files()
        if not files:
            markdown = self._build_context_markdown([])
            self.write_context(markdown)
            return markdown

        if not self.groq_client:
            markdown = self._build_context_fallback(files)
            self.write_context(markdown)
            return markdown

        all_summaries: list[FileSummary] = []
        for i, file_path in enumerate(files):
            summary = self._summarize_with_retry(file_path)
            all_summaries.append(summary)
            if progress_callback:
                progress_callback(i + 1, len(files))

        markdown = self._build_context_markdown(all_summaries)
        self.write_context(markdown)
        return markdown

    async def index_workspace_async(
        self, progress_callback: Callable[[int, int], None] | None = None
    ) -> str:
        """Asynchronously scan workspace files and generate context.md."""
        files = self._scan_files()
        if not files:
            markdown = self._build_context_markdown([])
            self.write_context(markdown)
            return markdown

        if not self.groq_client:
            markdown = self._build_context_fallback(files)
            self.write_context(markdown)
            return markdown

        all_summaries: list[FileSummary] = []
        for i, file_path in enumerate(files):
            summary = await self._summarize_with_retry_async(file_path)
            all_summaries.append(summary)
            if progress_callback:
                progress_callback(i + 1, len(files))

        markdown = self._build_context_markdown(all_summaries)
        self.write_context(markdown)
        return markdown

    def update_file(self, file_path: Path | str) -> None:
        """Update context.md for a specific file that changed."""
        path_obj = Path(file_path)
        if not path_obj.is_absolute():
            path_obj = self.workspace_root / path_obj

        if not path_obj.exists():
            return

        rel_path = str(path_obj.relative_to(self.workspace_root))
        summary = self._summarize_file(path_obj)

        current_context = self.read_context()
        if not current_context:
            self.index_workspace()
            return

        section_header = f"### `{rel_path}`"
        new_section = self._format_file_section(summary)

        if section_header in current_context:
            start_idx = current_context.index(section_header)
            after_header = current_context[start_idx + len(section_header) :]
            next_match = re.search(r"\n(#{1,3} )", after_header)
            if next_match:
                end_idx = start_idx + len(section_header) + next_match.start()
                updated = (
                    current_context[:start_idx]
                    + new_section
                    + current_context[end_idx:]
                )
            else:
                updated = current_context[:start_idx] + new_section
        else:
            updated = current_context.rstrip() + "\n\n" + new_section

        self.write_context(updated)

    # ─── Summarization & Retries ──────────────────────────────────────────────

    def _summarize_with_retry(
        self, abs_path: Path, max_retries: int = 3
    ) -> FileSummary:
        last_err: Exception | None = None
        for attempt in range(max_retries):
            try:
                return self._summarize_file(abs_path)
            except Exception as err:
                last_err = err
                msg = str(err)
                if "429" in msg or "rate_limit_exceeded" in msg:
                    delay_match = re.search(
                        r"try again in\s+([\d.]+)s", msg, re.IGNORECASE
                    )
                    delay = (
                        float(delay_match.group(1))
                        if delay_match
                        else (2.0**attempt) * 2.0
                    )
                    time.sleep(delay)
                    continue
                raise err

        rel_path = str(abs_path.relative_to(self.workspace_root))
        return FileSummary(
            relative_path=rel_path,
            purpose=f"Rate-limited after {max_retries} retries: {str(last_err)[:120]}",
        )

    async def _summarize_with_retry_async(
        self, abs_path: Path, max_retries: int = 3
    ) -> FileSummary:
        last_err: Exception | None = None
        for attempt in range(max_retries):
            try:
                return await self._summarize_file_async(abs_path)
            except Exception as err:
                last_err = err
                msg = str(err)
                if "429" in msg or "rate_limit_exceeded" in msg:
                    delay_match = re.search(
                        r"try again in\s+([\d.]+)s", msg, re.IGNORECASE
                    )
                    delay = (
                        float(delay_match.group(1))
                        if delay_match
                        else (2.0**attempt) * 2.0
                    )
                    await asyncio.sleep(delay)
                    continue
                raise err

        rel_path = str(abs_path.relative_to(self.workspace_root))
        return FileSummary(
            relative_path=rel_path,
            purpose=f"Rate-limited after {max_retries} retries: {str(last_err)[:120]}",
        )

    def _summarize_file(self, abs_path: Path) -> FileSummary:
        rel_path = str(abs_path.relative_to(self.workspace_root))
        max_file_size = 30 * 1024

        try:
            if abs_path.stat().st_size > max_file_size:
                return FileSummary(
                    relative_path=rel_path,
                    purpose=f"Large file ({(abs_path.stat().st_size / 1024):.0f}KB), skipped for LLM summarization.",
                )
            content = abs_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return FileSummary(relative_path=rel_path, purpose="Could not read file.")

        lines = content.splitlines()
        truncated = "\n".join(lines[:200])

        system_prompt = (
            "You are a code analyzer. Given a source file, produce a concise structured summary.\n"
            "Respond ONLY with a JSON object in this exact format:\n"
            '{ "purpose": "one-sentence description", "keyElements": ["func1", "class2"], "dependencies": ["module1"] }\n'
            "Keep purpose under 100 characters. List at most 8 key elements and 5 dependencies."
        )

        user_prompt = (
            f"File: {rel_path}\n```\n{truncated}\n```\nGenerate the JSON summary."
        )

        try:
            if self.groq_client:
                response = self.groq_client.complete(
                    "llama-3.1-8b-instant",
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.1,
                )
                match = re.search(r"\{[\s\S]*\}", response)
                if match:
                    parsed = json.loads(match.group(0))
                    return FileSummary(
                        relative_path=rel_path,
                        purpose=parsed.get("purpose", "No description provided."),
                        key_elements=parsed.get("keyElements", []),
                        dependencies=parsed.get("dependencies", []),
                    )
        except Exception as err:
            logger.debug(f"[ContextEngine] LLM summary failed for {rel_path}: {err}")

        return FileSummary(
            relative_path=rel_path,
            purpose=f"Source file ({len(lines)} lines).",
            key_elements=self._extract_top_level_names(truncated),
            dependencies=self._extract_imports(truncated),
        )

    async def _summarize_file_async(self, abs_path: Path) -> FileSummary:
        rel_path = str(abs_path.relative_to(self.workspace_root))
        max_file_size = 30 * 1024

        try:
            if abs_path.stat().st_size > max_file_size:
                return FileSummary(
                    relative_path=rel_path,
                    purpose=f"Large file ({(abs_path.stat().st_size / 1024):.0f}KB), skipped for LLM summarization.",
                )
            content = abs_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return FileSummary(relative_path=rel_path, purpose="Could not read file.")

        lines = content.splitlines()
        truncated = "\n".join(lines[:200])

        system_prompt = (
            "You are a code analyzer. Given a source file, produce a concise structured summary.\n"
            "Respond ONLY with a JSON object in this exact format:\n"
            '{ "purpose": "one-sentence description", "keyElements": ["func1", "class2"], "dependencies": ["module1"] }\n'
            "Keep purpose under 100 characters. List at most 8 key elements and 5 dependencies."
        )

        user_prompt = (
            f"File: {rel_path}\n```\n{truncated}\n```\nGenerate the JSON summary."
        )

        try:
            if self.groq_client:
                response = await self.groq_client.complete_async(
                    "llama-3.1-8b-instant",
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.1,
                )
                match = re.search(r"\{[\s\S]*\}", response)
                if match:
                    parsed = json.loads(match.group(0))
                    return FileSummary(
                        relative_path=rel_path,
                        purpose=parsed.get("purpose", "No description provided."),
                        key_elements=parsed.get("keyElements", []),
                        dependencies=parsed.get("dependencies", []),
                    )
        except Exception as err:
            logger.debug(
                f"[ContextEngine] Async LLM summary failed for {rel_path}: {err}"
            )

        return FileSummary(
            relative_path=rel_path,
            purpose=f"Source file ({len(lines)} lines).",
            key_elements=self._extract_top_level_names(truncated),
            dependencies=self._extract_imports(truncated),
        )

    # ─── Fallback Regex Extractor ─────────────────────────────────────────────

    def _extract_top_level_names(self, text: str) -> list[str]:
        names: set[str] = set()

        # TypeScript / JavaScript / Python functions & classes
        for m in re.finditer(
            r"^(?:export\s+)?(?:async\s+)?function\s+(\w+)", text, re.MULTILINE
        ):
            names.add(m.group(1))
        for m in re.finditer(
            r"^(?:export\s+)?const\s+(\w+)\s*[:=]", text, re.MULTILINE
        ):
            names.add(m.group(1))
        for m in re.finditer(r"^(?:export\s+)?class\s+(\w+)", text, re.MULTILINE):
            names.add(m.group(1))
        for m in re.finditer(r"^(?:async\s+)?def\s+(\w+)", text, re.MULTILINE):
            names.add(m.group(1))

        return list(names)[:15]

    def _extract_imports(self, text: str) -> list[str]:
        deps: set[str] = set()

        for m in re.finditer(r'from\s+["\']([^"\']+)["\']', text):
            mod = m.group(1)
            if not mod.startswith("."):
                deps.add(mod.split("/")[0])
        for m in re.finditer(r'require\s*\(\s*["\']([^"\']+)["\']\s*\)', text):
            mod = m.group(1)
            if not mod.startswith("."):
                deps.add(mod.split("/")[0])
        for m in re.finditer(r"^(?:import|from)\s+(\w+)", text, re.MULTILINE):
            deps.add(m.group(1))

        return list(deps)[:10]

    # ─── Helpers ───────────────────────────────────────────────────────────────

    def _scan_files(self, max_files: int = 200) -> list[Path]:
        valid_exts = {
            ".ts",
            ".js",
            ".tsx",
            ".jsx",
            ".py",
            ".go",
            ".rs",
            ".md",
            ".json",
            ".yaml",
            ".yml",
            ".toml",
            ".cfg",
            ".ini",
            ".html",
            ".css",
        }

        matched: list[Path] = []
        for path in self.workspace_root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix not in valid_exts:
                continue

            rel_str = str(path.relative_to(self.workspace_root))
            if any(
                part in rel_str.split("/")
                for part in (
                    "node_modules",
                    ".git",
                    ".cache",
                    ".venv",
                    "__pycache__",
                    "venv",
                    "env",
                    "dist",
                    "out",
                    "build",
                )
            ):
                continue

            matched.append(path)
            if len(matched) >= max_files:
                break

        return matched

    def _build_context_fallback(self, file_paths: list[Path]) -> str:
        summaries: list[FileSummary] = []
        for path in file_paths:
            rel = str(path.relative_to(self.workspace_root))
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                content = ""
            summaries.append(
                FileSummary(
                    relative_path=rel,
                    purpose="Source file.",
                    key_elements=self._extract_top_level_names(content),
                    dependencies=self._extract_imports(content),
                )
            )

        return self._build_context_markdown(summaries)

    def _build_context_markdown(self, summaries: list[FileSummary]) -> str:
        now_iso = datetime.now(UTC).isoformat()
        lines: list[str] = [
            "# Workspace Context",
            "",
            f"Generated: {now_iso}",
            f"Files indexed: {len(summaries)}",
            "",
            "## File Structure",
            "",
        ]

        for summary in summaries:
            lines.append(f"- `{summary.relative_path}` — {summary.purpose}")

        lines.extend(["", "## Detailed Summaries", ""])

        for summary in summaries:
            lines.append(self._format_file_section(summary))

        return "\n".join(lines)

    def _format_file_section(self, summary: FileSummary) -> str:
        lines: list[str] = [
            f"### `{summary.relative_path}`",
            "",
            f"**Purpose:** {summary.purpose}",
            "",
        ]

        if summary.key_elements:
            lines.append("**Key elements:**")
            for el in summary.key_elements:
                lines.append(f"- `{el}`")
            lines.append("")

        if summary.dependencies:
            lines.append("**Dependencies:**")
            for dep in summary.dependencies:
                lines.append(f"- `{dep}`")
            lines.append("")

        return "\n".join(lines)
