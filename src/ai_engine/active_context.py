# Copyright (c) 2026 Frank1o3
# SPDX-License-Identifier: MIT

"""ActiveFileTracker — tracks active file,
cursor position, enclosing class, and function.

Provides scope parsing for Python, TypeScript,
JavaScript, C-style languages, Go, and Rust.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast


@dataclass(slots=True)
class ActiveContext:
    """Structured information about currently active editing position."""

    file_path: str
    file_name: str
    line: int
    column: int
    class_name: str | None
    function_name: str | None
    language: str


class ActiveFileTracker:
    """Tracks active file, line, enclosing class name, and function name."""

    def __init__(self, workspace_root: Path | str | None = None) -> None:
        self.workspace_root = Path(workspace_root) if workspace_root else Path.cwd()
        self.current_context: ActiveContext | None = None

    def update_context(
        self,
        file_path: Path | str,
        line: int,
        column: int = 1,
        content: str | None = None,
    ) -> ActiveContext:
        """Update active context for a file and line number."""
        path_obj = Path(file_path)
        if not path_obj.is_absolute():
            path_obj = self.workspace_root / path_obj

        if content is None:
            if path_obj.exists():
                try:
                    content = path_obj.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    content = ""
            else:
                content = ""

        lines = content.splitlines()
        lang = self._detect_language(path_obj)
        try:
            rel_name = str(path_obj.relative_to(self.workspace_root))
        except ValueError:
            rel_name = path_obj.name

        class_name = self.find_enclosing_class(lines, line - 1, lang)
        func_name = self.find_enclosing_function(lines, line - 1, lang)

        self.current_context = ActiveContext(
            file_path=str(path_obj),
            file_name=rel_name,
            line=line,
            column=column,
            class_name=class_name,
            function_name=func_name,
            language=lang,
        )
        return self.current_context

    def get_active_context_string(self) -> str | None:
        """Format active context as a human-readable prompt string for the AI."""
        if not self.current_context:
            return None
        ctx = self.current_context
        parts = [f"Currently editing: `{ctx.file_name}` (line {ctx.line})"]
        if ctx.class_name:
            parts.append(f"  Inside class: `{ctx.class_name}`")
        if ctx.function_name:
            parts.append(f"  Inside function: `{ctx.function_name}`")
        if ctx.language:
            parts.append(f"  Language: {ctx.language}")

        return "\n".join(parts)

    def get_compact_display(self) -> str | None:
        """Get compact one-line status string."""
        if not self.current_context:
            return None
        ctx = self.current_context
        scope = (
            f"{ctx.function_name}()"
            if ctx.function_name
            else f"class {ctx.class_name}"
            if ctx.class_name
            else ""
        )
        scope_part = f" → {scope}" if scope else ""
        return f"{ctx.file_name}{scope_part} :{ctx.line}"

    def find_enclosing_class(
        self, lines: list[str], target_line: int, language: str
    ) -> str | None:
        """Find enclosing class name above target line."""
        if language == "Python":
            class_pattern = re.compile(r"^\s*class\s+(\w+)")
            current_class: str | None = None
            class_indent = -1

            for i in range(min(target_line, len(lines))):
                line = lines[i]
                match = class_pattern.match(line)
                if match:
                    indent = len(line) - len(line.lstrip())
                    if class_indent == -1 or indent <= class_indent:
                        current_class = cast(str, match.group(1))
                        class_indent = indent
            return current_class

        # C-style / TypeScript / JavaScript / Java
        class_pattern = re.compile(
            r"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+(\w+)"
        )
        for i in range(
            min(target_line - 1, len(lines) - 1), max(-1, target_line - 100), -1
        ):
            match = class_pattern.match(lines[i])
            if match:
                return cast(str, match.group(1))

        return None

    def find_enclosing_function(
        self, lines: list[str], target_line: int, language: str
    ) -> str | None:
        """Find enclosing function name above target line."""
        if language == "Python":
            def_pattern = re.compile(r"^\s*(?:async\s+)?def\s+(\w+)\s*\(")
            current_func: str | None = None
            func_indent = -1

            for i in range(min(target_line, len(lines))):
                line = lines[i]
                match = def_pattern.match(line)
                if match:
                    indent = len(line) - len(line.lstrip())
                    if func_indent == -1 or indent <= func_indent:
                        current_func = cast(str, match.group(1))
                        func_indent = indent
            return current_func

        func_patterns = [
            re.compile(
                r"^\s*(?:export\s+)?(?:async\s+)?(?:function|const|let|var)\s+(\w+)"
            ),
            re.compile(
                r"^\s*(?:private|public|protected|static)\s+(?:async\s+)?(?:function\s+)?(\w+)\s*\("
            ),
            re.compile(r"^\s*(\w+)\s*\([^)]*\)\s*(?::\s*\w+\s*)?\{"),
            re.compile(r"^\s*fn\s+(\w+)"),  # Rust
            re.compile(r"^\s*func\s+(?:\([^)]+\)\s+)?(\w+)"),  # Go
        ]

        for i in range(
            min(target_line, len(lines) - 1), max(-1, target_line - 100), -1
        ):
            line = lines[i]
            for pattern in func_patterns:
                match = pattern.match(line)
                if match:
                    return cast(str, match.group(1))

        return None

    def _detect_language(self, path: Path) -> str:
        ext = path.suffix.lower()
        mapping = {
            ".py": "Python",
            ".ts": "TypeScript",
            ".tsx": "TypeScript",
            ".js": "JavaScript",
            ".jsx": "JavaScript",
            ".go": "Go",
            ".rs": "Rust",
            ".java": "Java",
            ".cpp": "C++",
            ".c": "C",
            ".cs": "C#",
            ".html": "HTML",
            ".css": "CSS",
            ".json": "JSON",
            ".md": "Markdown",
        }
        return mapping.get(ext, "Plain Text")
