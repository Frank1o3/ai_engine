# Copyright (c) 2026 Frank1o3
# SPDX-License-Identifier: MIT

"""CLI entrypoint for GropWave AI Engine.

Usage:
  ai-engine index [--workspace PATH]
  ai-engine prompt "Your question" [--model MODEL_ID] [--workspace PATH]
  ai-engine quota
  ai-engine memory query "search query"
  ai-engine chat
"""

import argparse
import sys
from pathlib import Path

from .context_engine import ContextEngine
from .groq_client import GroqClient
from .memory_engine import MemoryEngine
from .orchestrator import OrchestratorCore
from .types import OrchestratorConfig


def main() -> None:
    """CLI entrypoint for ai-engine."""
    parser = argparse.ArgumentParser(description="GropWave AI Engine CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Index command
    index_parser = subparsers.add_parser(
        "index", help="Index workspace and generate context.md"
    )
    index_parser.add_argument(
        "--workspace", type=str, default=".", help="Workspace root directory"
    )

    # Prompt command
    prompt_parser = subparsers.add_parser(
        "prompt", help="Dispatch a prompt to AI Orchestrator"
    )
    prompt_parser.add_argument("text", type=str, help="Prompt text")
    prompt_parser.add_argument(
        "--model", type=str, default="auto", help="Model ID or 'auto'"
    )
    prompt_parser.add_argument(
        "--workspace", type=str, default=".", help="Workspace root directory"
    )
    prompt_parser.add_argument(
        "--stream", action="store_true", help="Stream response tokens"
    )

    # Quota command
    subparsers.add_parser("quota", help="Display quota health and limit statuses")

    # Memory command
    memory_parser = subparsers.add_parser("memory", help="Memory store utilities")
    memory_sub = memory_parser.add_subparsers(dest="memory_command")
    query_parser = memory_sub.add_parser("query", help="Query memory entries")
    query_parser.add_argument("query_text", type=str, help="Search text")
    query_parser.add_argument(
        "--workspace", type=str, default=".", help="Workspace root directory"
    )

    # Interactive chat command
    chat_parser = subparsers.add_parser(
        "chat", help="Start interactive CLI chat session"
    )
    chat_parser.add_argument(
        "--workspace", type=str, default=".", help="Workspace root directory"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    config = OrchestratorConfig()

    if args.command == "index":
        root = Path(args.workspace).resolve()
        client = GroqClient(config)
        ctx_engine = ContextEngine(config, client, root)
        print(f"Indexing workspace at {root}...")

        def on_progress(current: int, total: int) -> None:
            print(f"  [{current}/{total}] files summarized...", end="\r", flush=True)

        markdown = ctx_engine.index_workspace(on_progress)
        print("\nIndexing complete! context.md written.")

    elif args.command == "prompt":
        root = Path(args.workspace).resolve()
        config.default_model = args.model
        orchestrator = OrchestratorCore(config, root)
        orchestrator.initialize()

        print(f"Model: {orchestrator.selected_model} | Dispatching...")

        if args.stream:

            def on_chunk(chunk: str) -> None:
                print(chunk, end="", flush=True)

            response = orchestrator.dispatch(args.text, on_chunk=on_chunk)
            print()
        else:
            response = orchestrator.dispatch(args.text)
            print(response)

        print(f"\n[Used model: {orchestrator.last_used_model}]")

    elif args.command == "quota":
        orchestrator = OrchestratorCore(config)
        orchestrator.initialize()
        statuses = orchestrator.get_quota_statuses()

        print("=== Model Quota Health ===")
        for model_id, status in statuses.items():
            health_icon = (
                "🟢"
                if status.health == "healthy"
                else "🟡"
                if status.health == "warning"
                else "🔴"
            )
            print(
                f"{health_icon} {model_id:<40} {status.health:<10} "
                f"Constrained: {status.most_constrained.key} ({(status.most_constrained.ratio * 100):.1f}%)"
            )

    elif args.command == "memory":
        if args.memory_command == "query":
            root = Path(args.workspace).resolve()
            memory = MemoryEngine(workspace_root=root)
            memory.load()
            res = memory.retrieve_relevant(args.query_text)
            print(res or "No relevant memory entries found.")
        else:
            memory_parser.print_help()

    elif args.command == "chat":
        root = Path(args.workspace).resolve()
        orchestrator = OrchestratorCore(config, root)
        orchestrator.initialize()

        print("=== GropWave Interactive AI Chat (type 'exit' to quit) ===")
        while True:
            try:
                user_input = input("\nYou > ").strip()
                if not user_input or user_input.lower() in ("exit", "quit"):
                    break

                print("AI > ", end="", flush=True)
                orchestrator.dispatch(
                    user_input, on_chunk=lambda c: print(c, end="", flush=True)
                )
                print()
            except KeyboardInterrupt:
                break
            except Exception as err:
                print(f"\nError: {err}")


if __name__ == "__main__":
    main()
