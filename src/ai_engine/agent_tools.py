# Copyright (c) 2026 Frank1o3
# SPDX-License-Identifier: MIT

"""AgentTools — agentic capabilities for terminal execution,
safe file edits, and tool parsing.

Supports parsing:
  <tool:run_command>npm test</tool:run_command>
  <tool:edit_file path="src/index.py">...content...</tool:edit_file>
"""

import asyncio
import re
import subprocess
from pathlib import Path

from .types import FileEditResult, ParsedToolCall, TerminalResult

DANGEROUS_COMMAND_PATTERNS = [
    re.compile(r"\brm\s+(-rf?|--recursive)\b", re.IGNORECASE),
    re.compile(r"\bdd\s", re.IGNORECASE),
    re.compile(r"\bmkfs\.", re.IGNORECASE),
    re.compile(r"\bsudo\s+(rm|dd|mkfs|chmod|chown)\b", re.IGNORECASE),
    re.compile(r">/dev/sd", re.IGNORECASE),
    re.compile(r"\bshred\b", re.IGNORECASE),
]


def is_dangerous_command(command: str) -> bool:
    """Check if command matches dangerous/destructive execution patterns."""
    return any(pattern.search(command) for pattern in DANGEROUS_COMMAND_PATTERNS)


def run_command_sync(
    command: str,
    cwd: Path | str | None = None,
    timeout: float = 30.0,
) -> TerminalResult:
    """Synchronously execute a terminal command with safety checks and timeout."""
    if is_dangerous_command(command):
        return TerminalResult(
            command=command,
            exit_code=1,
            stdout="",
            stderr="Command blocked: matches dangerous pattern.",
            timed_out=False,
        )

    working_dir = str(cwd) if cwd else None
    try:
        process = subprocess.run(
            command,
            cwd=working_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        )
        return TerminalResult(
            command=command,
            exit_code=process.returncode,
            stdout=process.stdout,
            stderr=process.stderr,
            timed_out=False,
        )
    except subprocess.TimeoutExpired as exc:
        return TerminalResult(
            command=command,
            exit_code=124,
            stdout=exc.stdout or "" if isinstance(exc.stdout, str) else "",
            stderr=exc.stderr or "Command timed out."
            if isinstance(exc.stderr, str)
            else "Command timed out.",
            timed_out=True,
        )
    except Exception as err:
        return TerminalResult(
            command=command,
            exit_code=1,
            stdout="",
            stderr=str(err),
            timed_out=False,
        )


async def run_command_async(
    command: str,
    cwd: Path | str | None = None,
    timeout: float = 30.0,
) -> TerminalResult:
    """Asynchronously execute a terminal command."""
    if is_dangerous_command(command):
        return TerminalResult(
            command=command,
            exit_code=1,
            stdout="",
            stderr="Command blocked: matches dangerous pattern.",
            timed_out=False,
        )

    working_dir = str(cwd) if cwd else None
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=working_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
        return TerminalResult(
            command=command,
            exit_code=proc.returncode,
            stdout=stdout_bytes.decode(errors="replace"),
            stderr=stderr_bytes.decode(errors="replace"),
            timed_out=False,
        )
    except TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        return TerminalResult(
            command=command,
            exit_code=124,
            stdout="",
            stderr="Command timed out.",
            timed_out=True,
        )
    except Exception as err:
        return TerminalResult(
            command=command,
            exit_code=1,
            stdout="",
            stderr=str(err),
            timed_out=False,
        )


def edit_file_content(file_path: Path | str, new_content: str) -> FileEditResult:
    """Synchronously edit or create a file with new content."""
    path_obj = Path(file_path)
    try:
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        if path_obj.exists():
            existing = path_obj.read_text(encoding="utf-8", errors="replace")
            if existing == new_content:
                return FileEditResult(
                    success=True, message="No changes — file content is identical."
                )

        path_obj.write_text(new_content, encoding="utf-8")
        return FileEditResult(
            success=True, message=f"Successfully updated `{path_obj.name}`"
        )
    except Exception as err:
        return FileEditResult(success=False, message=f"Edit failed: {err}")


def parse_tool_calls(response: str) -> list[ParsedToolCall]:
    """Parse XML-style tool calls from LLM response text."""
    calls: list[ParsedToolCall] = []

    command_regex = re.compile(r"<tool:run_command>([\s\S]*?)</tool:run_command>")
    for match in command_regex.finditer(response):
        calls.append(ParsedToolCall(type="run_command", payload=match.group(1).strip()))

    edit_regex = re.compile(
        r'<tool:edit_file\s+path="([^"]+)">([\s\S]*?)</tool:edit_file>'
    )
    for match in edit_regex.finditer(response):
        calls.append(
            ParsedToolCall(
                type="edit_file",
                payload=match.group(2).strip(),
                target_path=match.group(1).strip(),
            )
        )

    return calls


def execute_tool_calls_sync(
    calls: list[ParsedToolCall], workspace_root: Path | str | None = None
) -> list[str]:
    """Synchronously execute parsed tool calls."""
    root = Path(workspace_root) if workspace_root else Path.cwd()
    results: list[str] = []

    for call in calls:
        if call.type == "run_command":
            if is_dangerous_command(call.payload):
                results.append(
                    f"Command blocked (dangerous): {call.payload}\n"
                    "Please confirm with user before running destructive commands."
                )
                continue
            res = run_command_sync(call.payload, cwd=root)
            results.append(
                f"Command: {call.payload}\nExit: {res.exit_code}\nStdout: {res.stdout}\nStderr: {res.stderr}"
            )

        elif call.type == "edit_file":
            if not call.target_path:
                results.append("Edit failed: no path provided")
                continue

            target_path = Path(call.target_path)
            if not target_path.is_absolute():
                target_path = root / target_path

            res_edit = edit_file_content(target_path, call.payload)
            results.append(f"Edit {call.target_path}: {res_edit.message}")

    return results


async def execute_tool_calls_async(
    calls: list[ParsedToolCall], workspace_root: Path | str | None = None
) -> list[str]:
    """Asynchronously execute parsed tool calls."""
    root = Path(workspace_root) if workspace_root else Path.cwd()
    results: list[str] = []

    for call in calls:
        if call.type == "run_command":
            if is_dangerous_command(call.payload):
                results.append(
                    f"Command blocked (dangerous): {call.payload}\n"
                    "Please confirm with user before running destructive commands."
                )
                continue
            res = await run_command_async(call.payload, cwd=root)
            results.append(
                f"Command: {call.payload}\nExit: {res.exit_code}\nStdout: {res.stdout}\nStderr: {res.stderr}"
            )

        elif call.type == "edit_file":
            if not call.target_path:
                results.append("Edit failed: no path provided")
                continue

            target_path = Path(call.target_path)
            if not target_path.is_absolute():
                target_path = root / target_path

            res_edit = edit_file_content(target_path, call.payload)
            results.append(f"Edit {call.target_path}: {res_edit.message}")

    return results
