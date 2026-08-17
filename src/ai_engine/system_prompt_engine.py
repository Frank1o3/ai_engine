# Copyright (c) 2026 Frank1o3
# SPDX-License-Identifier: MIT

"""SystemPromptEngine — manages system.md and performs selective feeding.

Extracts only the sections relevant to the user's current task instead of sending
the entire system.md file on every call.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .types import OrchestratorConfig

logger = logging.getLogger(__name__)

STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "shall",
    "can",
    "need",
    "dare",
    "ought",
    "used",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "at",
    "by",
    "from",
    "as",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "between",
    "out",
    "off",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "and",
    "but",
    "or",
    "nor",
    "not",
    "so",
    "yet",
    "both",
    "either",
    "neither",
    "each",
    "every",
    "all",
    "any",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "only",
    "own",
    "same",
    "than",
    "too",
    "very",
    "just",
    "because",
    "if",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "what",
    "how",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "we",
    "our",
    "you",
    "your",
    "they",
    "their",
    "he",
    "she",
    "his",
    "her",
    "i",
    "my",
    "me",
    "about",
    "also",
    "make",
    "like",
    "use",
    "using",
    "ensure",
    "follow",
    "always",
    "never",
    "prefer",
    "avoid",
    "note",
    "important",
    "must",
    "done",
}


def extract_keywords(text: str, max_count: int = 15) -> list[str]:
    """Extract representative keywords from a block of text, filtered by stopwords."""
    clean = re.sub(r"[^a-z0-9\s_\-+#./]", " ", text.lower())
    words = [w for w in clean.split() if len(w) > 2 and w not in STOPWORDS]

    # Sort longer words first as they are generally more distinctive
    words.sort(key=len, reverse=True)

    seen: set[str] = set()
    unique: list[str] = []
    for w in words:
        if w not in seen:
            seen.add(w)
            unique.append(w)
            if len(unique) >= max_count:
                break

    return unique


DEFAULT_SYSTEM_MD = """## General Rules

- Always read the user's request carefully and reference workspace context before responding.
- When writing code, prefer clear, idiomatic patterns appropriate for the target language.
- Prefer small, focused functions over large monolithic blocks.
- When unsure about a file's contents, inspect it using tool calls or request clarification.
- Never fabricate information about files you have not seen.
- When modifying code, use the `<tool:edit_file>` tool with the full updated content of the affected region.

## Code Style

- Use meaningful variable and function names. Avoid single-letter names except for loop indices.
- Add comments only when the intent is non-obvious; prefer self-documenting code.
- When generating TypeScript, use strict types — avoid `any` unless necessary.
- When generating Python, use PEP 8 conventions, type hints, and modern 3.14 syntax (`X | Y`, `list[T]`).
- Keep imports grouped logically: standard library, third-party, then local modules.

## Testing Standards

- When asked to write tests, match the project's existing test framework (pytest, unittest, etc.).
- Write tests covering happy paths and relevant edge cases.
- Use descriptive test names explaining what is verified.

## Terminal Usage

- When running commands, use the `<tool:run_command>` tool.
- Prefer non-destructive commands. Avoid `rm -rf`, `dd`, or similar without explicit confirmation.
- Check the project package manager (uv, poetry, pip, npm) before running installation commands.
""".strip()


@dataclass(slots=True)
class Section:
    """Parsed section of system.md."""

    heading: str
    content: str
    keywords: list[str]


class SystemPromptEngine:
    """Manages system instructions in system.md and selective prompt extraction."""

    def __init__(
        self, config: OrchestratorConfig, workspace_root: Path | str | None = None
    ) -> None:
        self.config = config
        self.workspace_root = Path(workspace_root) if workspace_root else Path.cwd()
        self._sections: list[Section] = []
        self._raw_content: str = ""
        self._generated_project_information: str = ""

    @classmethod
    def ensure_default_exists(
        cls, config: OrchestratorConfig, workspace_root: Path | str | None = None
    ) -> bool:
        """Create a default system.md in workspace root if it does not exist."""
        root = Path(workspace_root) if workspace_root else Path.cwd()
        system_path = root / config.cache_directory / config.system_file
        system_path.parent.mkdir(parents=True, exist_ok=True)
        if not system_path.exists():
            system_path.write_text(DEFAULT_SYSTEM_MD, encoding="utf-8")
            return True
        return False

    def get_system_path(self) -> Path:
        """Get absolute path to system.md."""
        return (
            self.workspace_root / self.config.cache_directory / self.config.system_file
        )

    def load(self) -> bool:
        """Read and parse system.md from workspace. Returns True if successfully parsed."""
        system_path = self.get_system_path()
        if not system_path.exists():
            self._sections = []
            self._raw_content = ""
            return False

        try:
            self._raw_content = system_path.read_text(encoding="utf-8")
        except Exception as err:
            logger.warning(
                "[SystemPromptEngine] Failed to read %s: %s", system_path, err
            )
            self._sections = []
            self._raw_content = ""
            return False

        self._sections = self._parse_sections(self._raw_content)
        return True

    def reload(self) -> bool:
        """Reload system.md from disk."""
        return self.load()

    def set_generated_project_information(self, facts: dict[str, str]) -> None:
        """Set generated facts without changing human-authored system instructions."""
        useful = [
            (key.replace("_", " ").title(), value)
            for key, value in facts.items()
            if value and value != "unknown"
        ]
        self._generated_project_information = (
            "\n".join([
                "## Generated Project Information",
                *[f"- {key}: {value}" for key, value in useful],
            ])
            if useful
            else ""
        )

    def extract_relevant(self, prompt: str, context_content: str | None = None) -> str:
        """Extract sections relevant to prompt and context based on keyword overlap scoring."""
        if not self._sections:
            return self._raw_content

        query_words: set[str] = set(extract_keywords(prompt, 30))
        if context_content:
            query_words.update(extract_keywords(context_content, 20))

        scored: list[tuple[Section, float]] = []
        for section in self._sections:
            overlap = 0.0
            for kw in section.keywords:
                if kw in query_words:
                    overlap += 1.0

            content_lower = section.content.lower()
            for qw in query_words:
                if qw in content_lower:
                    overlap += 0.5

            scored.append((section, overlap))

        general_section = next(
            (
                s
                for s in self._sections
                if re.search(
                    r"general|rules?|guidelines?|standards?", s.heading, re.IGNORECASE
                )
            ),
            None,
        )

        relevant = [s for s, score in scored if score > 0.0]

        if general_section and general_section not in relevant:
            relevant.insert(0, general_section)

        if not relevant and self._sections:
            relevant.append(self._sections[0])

        selected = "\n\n".join(f"## {s.heading}\n{s.content.strip()}" for s in relevant)
        return "\n\n".join(
            part for part in (selected, self._generated_project_information) if part
        )

    def get_all_sections(self) -> list[Section]:
        """Return all parsed sections."""
        return self._sections

    def get_raw_content(self) -> str:
        """Return raw system.md text."""
        return self._raw_content

    def _parse_sections(self, content: str) -> list[Section]:
        lines = content.splitlines()
        sections: list[Section] = []
        current_heading = ""
        current_lines: list[str] = []

        def flush():
            nonlocal current_heading, current_lines
            if current_heading and current_lines:
                text = "\n".join(current_lines).strip()
                sections.append(
                    Section(
                        heading=current_heading,
                        content=text,
                        keywords=extract_keywords(text),
                    )
                )

        for line in lines:
            match = re.match(r"^##\s+(.+)$", line)
            if match:
                flush()
                current_heading = match.group(1).strip()
                current_lines = []
            else:
                current_lines.append(line)
        flush()

        return sections
