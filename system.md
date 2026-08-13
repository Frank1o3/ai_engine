# General Rules

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

