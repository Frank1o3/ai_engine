# Copyright (c) 2026 Frank1o3
# SPDX-License-Identifier: MIT

"""Incremental workspace indexing backed by a project-local SQLite database.

The generated context.md is intentionally a compact, human-readable map. Rich
file facts are queried from SQLite per prompt instead of being injected whole.
"""

import ast
import hashlib
import sqlite3
import tomllib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from .system_prompt_engine import extract_keywords

if TYPE_CHECKING:
    from collections.abc import Callable

    from .groq_client import GroqClient
    from .types import OrchestratorConfig


CONTEXT_VERSION = 2
EXCLUDED_PARTS = frozenset({
    ".git",
    ".groqwave_cache",
    ".venv",
    "__pycache__",
    "node_modules",
    "venv",
    "env",
    "dist",
    "out",
    "build",
})
SOURCE_EXTENSIONS = {
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".go",
    ".rs",
    ".java",
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


@dataclass(slots=True)
class FileSummary:
    """Compatibility view of indexed file data."""

    relative_path: str
    purpose: str
    key_elements: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)


@dataclass(slots=True)
class IndexedSymbol:
    name: str
    kind: str
    signature: str
    line: int
    column: int
    enclosing_class: str | None = None


class ContextEngine:
    """Maintain an idempotent incremental index and retrieve focused context."""

    def __init__(
        self,
        config: OrchestratorConfig,
        groq_client: GroqClient | None = None,
        workspace_root: Path | str | None = None,
    ) -> None:
        self.config = config
        self.groq_client = groq_client  # Future optional richer summaries.
        self.workspace_root = Path(workspace_root) if workspace_root else Path.cwd()
        self.cache_dir = self.workspace_root / config.cache_directory
        self.db_path = self.cache_dir / "workspace.db"

    def get_context_path(self) -> Path:
        return self.cache_dir / self.config.context_file

    def read_context(self) -> str:
        try:
            return self.get_context_path().read_text(encoding="utf-8")
        except OSError:
            return ""

    def write_context(self, content: str) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.get_context_path().write_text(content, encoding="utf-8")

    def index_workspace(
        self, progress_callback: Callable[[int, int], None] | None = None
    ) -> str:
        """Hash files, changing only affected database rows, then refresh context.md."""
        files = self._scan_files()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            self._ensure_schema(db)
            known = {
                r["path"]: r["hash"] for r in db.execute("SELECT path, hash FROM files")
            }
            present: set[str] = set()
            for number, path in enumerate(files, 1):
                rel = path.relative_to(self.workspace_root).as_posix()
                present.add(rel)
                content = path.read_text(encoding="utf-8", errors="replace")
                digest = hashlib.sha256(content.encode()).hexdigest()
                if known.get(rel) != digest:
                    self._index_file(db, rel, path, content, digest)
                if progress_callback:
                    progress_callback(number, len(files))
            db.executemany(
                "DELETE FROM files WHERE path = ?",
                ((path,) for path in set(known) - present),
            )
            self._refresh_project_metadata(db)
            markdown = self._build_context_markdown(db)
        self.write_context(markdown)
        return markdown

    async def index_workspace_async(
        self, progress_callback: Callable[[int, int], None] | None = None
    ) -> str:
        """Async-compatible API; local deterministic indexing needs no remote calls."""
        return self.index_workspace(progress_callback)

    def update_file(self, file_path: Path | str) -> None:
        """Apply a single-file forward patch; deleted files remove dependent records."""
        path = Path(file_path)
        path = path if path.is_absolute() else self.workspace_root / path
        try:
            rel = path.relative_to(self.workspace_root).as_posix()
        except ValueError:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            self._ensure_schema(db)
            if path.is_file() and self._is_indexable(path):
                content = path.read_text(encoding="utf-8", errors="replace")
                digest = hashlib.sha256(content.encode()).hexdigest()
                old = db.execute(
                    "SELECT hash FROM files WHERE path = ?", (rel,)
                ).fetchone()
                if old is None or old["hash"] != digest:
                    self._index_file(db, rel, path, content, digest)
            else:
                db.execute("DELETE FROM files WHERE path = ?", (rel,))
            self._refresh_project_metadata(db)
            markdown = self._build_context_markdown(db)
        self.write_context(markdown)

    def retrieve_relevant(self, prompt: str, max_files: int = 6) -> str:
        """Return compact indexed facts for the most relevant files and symbols."""
        if not self.db_path.exists():
            return ""
        words = extract_keywords(prompt, 20)
        if not words:
            return ""
        score_terms = " + ".join(
            "CASE WHEN lower(path) LIKE ? THEN 4 "
            "WHEN lower(context) LIKE ? THEN 1 ELSE 0 END + "
            "CASE WHEN EXISTS (SELECT 1 FROM symbols s "
            "WHERE s.file_id = files.id AND lower(s.name) LIKE ?) THEN 6 ELSE 0 END"
            for _ in words
        )
        params = [f"%{word}%" for word in words for _ in range(3)]
        with self._connect() as db:
            self._ensure_schema(db)
            rows = db.execute(
                f"SELECT * FROM (SELECT *, ({score_terms}) AS relevance FROM files) "
                "WHERE relevance > 0 ORDER BY relevance DESC, path LIMIT ?",
                (*params, max_files),
            ).fetchall()
            blocks: list[str] = []
            for row in rows:
                symbols = db.execute(
                    "SELECT name,kind,signature,line,column,enclosing_class FROM symbols WHERE file_id = ? ORDER BY line LIMIT 20",
                    (row["id"],),
                ).fetchall()
                lines = [
                    f"### {row['path']}",
                    f"{row['language']}; {row['line_count']} lines; sha256 {row['hash'][:12]}",
                    row["context"],
                ]
                for symbol in symbols:
                    qualified = ".".join(
                        filter(None, (symbol["enclosing_class"], symbol["name"]))
                    )
                    lines.append(
                        f"- {symbol['kind']} `{qualified}` — line {symbol['line']}, column {symbol['column']}: `{symbol['signature']}`"
                    )
                blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    def find_symbol(self, name: str) -> list[IndexedSymbol]:
        if not self.db_path.exists():
            return []
        with self._connect() as db:
            rows = db.execute(
                "SELECT name,kind,signature,line,column,enclosing_class FROM symbols WHERE name = ? OR enclosing_class || '.' || name = ? ORDER BY line",
                (name.rsplit(".", 1)[-1], name),
            ).fetchall()
        return [IndexedSymbol(**dict(row)) for row in rows]

    def get_project_profile(self) -> dict[str, str]:
        """Return lightweight generated facts for the system prompt layer."""
        if not self.db_path.exists():
            return {}
        with self._connect() as db:
            self._ensure_schema(db)
            return dict(db.execute("SELECT key, value FROM project_metadata"))

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.db_path)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        return db

    @staticmethod
    def _ensure_schema(db: sqlite3.Connection) -> None:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS files (id INTEGER PRIMARY KEY, path TEXT UNIQUE NOT NULL, hash TEXT NOT NULL, language TEXT NOT NULL, line_count INTEGER NOT NULL, class_count INTEGER NOT NULL, function_count INTEGER NOT NULL, symbol_count INTEGER NOT NULL, context TEXT NOT NULL, indexed_at TEXT NOT NULL, context_version INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS symbols (id INTEGER PRIMARY KEY, file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE, name TEXT NOT NULL, kind TEXT NOT NULL, signature TEXT NOT NULL, line INTEGER NOT NULL, column INTEGER NOT NULL, enclosing_class TEXT);
            CREATE INDEX IF NOT EXISTS symbols_name_idx ON symbols(name);
            CREATE TABLE IF NOT EXISTS dependencies (source_file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE, target TEXT NOT NULL, dependency_type TEXT NOT NULL, PRIMARY KEY(source_file_id,target,dependency_type));
            CREATE TABLE IF NOT EXISTS project_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """)

    def _index_file(
        self, db: sqlite3.Connection, rel: str, path: Path, content: str, digest: str
    ) -> None:
        language = self._language(path)
        symbols, imports = self._analyze(content, language)
        context = self._describe(language, len(content.splitlines()), symbols, imports)
        old = db.execute("SELECT id FROM files WHERE path = ?", (rel,)).fetchone()
        if old:
            db.execute("DELETE FROM files WHERE id = ?", (old["id"],))
        file_id = db.execute(
            "INSERT INTO files(path,hash,language,line_count,class_count,function_count,symbol_count,context,indexed_at,context_version) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                rel,
                digest,
                language,
                len(content.splitlines()),
                sum(s.kind == "class" for s in symbols),
                sum(s.kind in {"function", "method"} for s in symbols),
                len(symbols),
                context,
                datetime.now(UTC).isoformat(),
                CONTEXT_VERSION,
            ),
        ).lastrowid
        db.executemany(
            "INSERT INTO symbols(file_id,name,kind,signature,line,column,enclosing_class) VALUES(?,?,?,?,?,?,?)",
            (
                (
                    file_id,
                    s.name,
                    s.kind,
                    s.signature,
                    s.line,
                    s.column,
                    s.enclosing_class,
                )
                for s in symbols
            ),
        )
        db.executemany(
            "INSERT INTO dependencies(source_file_id,target,dependency_type) VALUES(?,?,?)",
            ((file_id, target, kind) for target, kind in imports),
        )

    def _analyze(
        self, content: str, language: str
    ) -> tuple[list[IndexedSymbol], list[tuple[str, str]]]:
        if language != "Python":
            return [], []
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return [], []
        symbols: list[IndexedSymbol] = []
        imports: list[tuple[str, str]] = []
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom):
                    imports.append((
                        node.module or ".",
                        "internal" if node.level else "external",
                    ))
                else:
                    imports.extend((alias.name, "external") for alias in node.names)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbols.append(self._python_symbol(node, "function"))
            elif isinstance(node, ast.ClassDef):
                symbols.append(
                    IndexedSymbol(
                        node.name,
                        "class",
                        f"class {node.name}",
                        node.lineno,
                        node.col_offset + 1,
                    )
                )
                symbols.extend(
                    self._python_symbol(child, "method", node.name)
                    for child in node.body
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                )
        return symbols, imports

    @staticmethod
    def _python_symbol(
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        kind: str,
        enclosing_class: str | None = None,
    ) -> IndexedSymbol:
        args = [arg.arg for arg in node.args.posonlyargs + node.args.args]
        if node.args.vararg:
            args.append("*" + node.args.vararg.arg)
        args.extend(arg.arg for arg in node.args.kwonlyargs)
        if node.args.kwarg:
            args.append("**" + node.args.kwarg.arg)
        result = f"{'async ' if isinstance(node, ast.AsyncFunctionDef) else ''}{node.name}({', '.join(args)})"
        if node.returns:
            result += f" -> {ast.unparse(node.returns)}"
        return IndexedSymbol(
            node.name, kind, result, node.lineno, node.col_offset + 1, enclosing_class
        )

    def _scan_files(self, max_files: int = 5000) -> list[Path]:
        return [
            path
            for path in sorted(self.workspace_root.rglob("*"))
            if path.is_file() and self._is_indexable(path)
        ][:max_files]

    def _is_indexable(self, path: Path) -> bool:
        try:
            relative = path.relative_to(self.workspace_root)
        except ValueError:
            return False
        return path.suffix.lower() in SOURCE_EXTENSIONS and not any(
            part in EXCLUDED_PARTS for part in relative.parts
        )

    @staticmethod
    def _language(path: Path) -> str:
        return {
            ".py": "Python",
            ".ts": "TypeScript",
            ".tsx": "TypeScript",
            ".js": "JavaScript",
            ".jsx": "JavaScript",
            ".go": "Go",
            ".rs": "Rust",
            ".java": "Java",
        }.get(path.suffix.lower(), "Text")

    @staticmethod
    def _describe(
        language: str,
        line_count: int,
        symbols: list[IndexedSymbol],
        imports: list[tuple[str, str]],
    ) -> str:
        names = (
            ", ".join(symbol.name for symbol in symbols[:8]) or "no extracted symbols"
        )
        dependencies = ", ".join(target for target, _ in imports[:6]) or "no imports"
        return f"{language} file with {line_count} lines. Symbols: {names}. Imports: {dependencies}."

    def _refresh_project_metadata(self, db: sqlite3.Connection) -> None:
        build = "unknown"
        values: dict[str, str] = {}
        pyproject = self.workspace_root / "pyproject.toml"
        if pyproject.exists():
            content = pyproject.read_bytes()
            data = tomllib.loads(content.decode("utf-8", errors="replace"))
            project = data.get("project", {})
            dependencies = project.get("dependencies", [])
            if dependencies:
                values["external_dependencies"] = ", ".join(
                    dependency.split("=", 1)[0].split(">", 1)[0]
                    for dependency in dependencies
                )
            if requires_python := project.get("requires-python"):
                values["python_version"] = requires_python
            build = (
                "uv"
                if "uv" in content.decode("utf-8", errors="replace").lower()
                else "pyproject.toml"
            )
        language = (
            "Python"
            if any(path.suffix == ".py" for path in self._scan_files())
            else "unknown"
        )
        values.update({"primary_language": language, "build_system": build})
        db.executemany(
            "INSERT OR REPLACE INTO project_metadata(key,value) VALUES(?,?)",
            values.items(),
        )

    @staticmethod
    def _build_context_markdown(db: sqlite3.Connection) -> str:
        rows = db.execute(
            "SELECT path,hash,language,line_count,symbol_count FROM files ORDER BY path"
        ).fetchall()
        metadata = dict(db.execute("SELECT key,value FROM project_metadata"))
        lines = [
            "# GropWave Workspace Context",
            "",
            "This is a lightweight workspace index. Detailed file context is stored in `.groqwave_cache/workspace.db`.",
            "",
            f"Generated: {datetime.now(UTC).isoformat()}",
            f"Files indexed: {len(rows)}",
            f"Primary language: {metadata.get('primary_language', 'unknown')}",
            f"Build system: {metadata.get('build_system', 'unknown')}",
            "",
            "## Files",
            "",
        ]
        if metadata.get("external_dependencies"):
            lines.insert(
                -2, f"External dependencies: {metadata['external_dependencies']}"
            )
        for row in rows:
            lines += [
                f"### {row['path']}",
                f"- {row['language']}; {row['line_count']} lines; {row['symbol_count']} symbols; sha256 `{row['hash'][:12]}`",
            ]
        return "\n".join(lines) + "\n"
