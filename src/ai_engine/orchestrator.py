# Copyright (c) 2026 Frank1o3
# SPDX-License-Identifier: MIT

"""OrchestratorCore — central coordinator for model selection,
routing, context feeding, and dispatch.

Coordinates:
  1. Prompt task classification via TaskRouter.
  2. Workspace & active file context assembly.
  3. System prompt selective section extraction via SystemPromptEngine.
  4. Memory retrieval via MemoryEngine.
  5. Quota health check and smart model routing with fallback tier chains.
  6. Dispatch via GroqClient (blocking or streaming).
  7. Automated tool execution loop for <tool:...> calls.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .active_context import ActiveFileTracker
from .agent_tools import (
    execute_tool_calls_async,
    execute_tool_calls_sync,
    parse_tool_calls,
)
from .context_engine import ContextEngine
from .groq_client import GroqClient
from .memory_engine import MemoryEngine
from .models import ModelRegistry
from .quota import QuotaTracker
from .router import TaskRouter, estimate_tokens
from .system_prompt_engine import SystemPromptEngine
from .types import (
    ModelQuotaStatus,
    OrchestratorConfig,
    RegisteredModel,
    TaskTier,
)

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

TIER_FALLBACK_ORDER: dict[TaskTier, list[TaskTier]] = {
    TaskTier.HEAVY: [TaskTier.HEAVY, TaskTier.BALANCED, TaskTier.FAST],
    TaskTier.BALANCED: [TaskTier.BALANCED, TaskTier.HEAVY, TaskTier.FAST],
    TaskTier.FAST: [TaskTier.FAST, TaskTier.BALANCED, TaskTier.HEAVY],
}

TOOL_PREAMBLE = """You have access to the following tools. Use them when appropriate to take action.

### Tools

**Run a terminal command:**
Wrap the command in `<tool:run_command>...</tool:run_command>` tags.
Example: `<tool:run_command>pytest</tool:run_command>`

**Edit a file:**
Wrap the new file content in `<tool:edit_file path="relative/path">...</tool:edit_file>` tags.
Example: `<tool:edit_file path="src/index.py">export const x = 1;</tool:edit_file>`

After using tools, summarize what you did and provide the final answer to the user.
Only use tools when they are genuinely needed — prefer answering from knowledge when possible.
"""


class OrchestratorCore:
    """The central AI engine orchestrator."""

    def __init__(
        self,
        config: OrchestratorConfig,
        workspace_root: Path | str | None = None,
        groq_client: GroqClient | None = None,
        model_registry: ModelRegistry | None = None,
        quota_tracker: QuotaTracker | None = None,
        task_router: TaskRouter | None = None,
        system_prompt_engine: SystemPromptEngine | None = None,
        context_engine: ContextEngine | None = None,
        memory_engine: MemoryEngine | None = None,
        active_file_tracker: ActiveFileTracker | None = None,
    ) -> None:
        self.config = config
        self.workspace_root = Path(workspace_root) if workspace_root else Path.cwd()

        self.groq_client = groq_client or GroqClient(config)
        self.model_registry = model_registry or ModelRegistry(self.groq_client.raw_sync)
        self.quota_tracker = quota_tracker or QuotaTracker(
            config.quota_warning_threshold
        )
        self.task_router = task_router or TaskRouter()

        self.system_prompt_engine = system_prompt_engine or SystemPromptEngine(
            config, self.workspace_root
        )
        self.context_engine = context_engine or ContextEngine(
            config, self.groq_client, self.workspace_root
        )
        self.memory_engine = memory_engine or MemoryEngine(
            config.memory_file, self.workspace_root
        )
        self.active_file_tracker = active_file_tracker or ActiveFileTracker(
            self.workspace_root
        )

        self.selected_model: str = config.default_model
        self.last_used_model: str = ""
        self._initialized = False

    def initialize(self) -> None:
        """Initialize synchronously: load system prompt, memory, and fetch model registry."""
        SystemPromptEngine.ensure_default_exists(self.config, self.workspace_root)
        self.system_prompt_engine.load()
        self.memory_engine.load()
        self.memory_engine.decay_all()
        self.model_registry.refresh()
        self._initialized = True

    async def initialize_async(self) -> None:
        """Initialize asynchronously."""
        SystemPromptEngine.ensure_default_exists(self.config, self.workspace_root)
        self.system_prompt_engine.load()
        self.memory_engine.load()
        self.memory_engine.decay_all()
        await self.model_registry.refresh_async()
        self._initialized = True

    def set_selected_model(self, model_id: str) -> None:
        """Set manually selected model or 'auto' for smart routing."""
        self.selected_model = model_id

    def get_models(self) -> list[RegisteredModel]:
        """Return registered model list."""
        return self.model_registry.get_models()

    def get_quota_statuses(self) -> dict[str, ModelQuotaStatus]:
        """Get current quota health for all registered models."""
        models = self.model_registry.get_models()
        limits_map = {m.id: m.limits for m in models}
        return self.quota_tracker.get_all_health(limits_map)

    # ─── Synchronous Dispatch ──────────────────────────────────────────────────

    def dispatch(
        self,
        prompt: str,
        on_chunk: Callable[[str], None] | None = None,
    ) -> str:
        """Synchronously process a user prompt."""
        if not self._initialized:
            self.initialize()

        task = self.task_router.classify(prompt)
        messages = self._build_messages(prompt)

        model_id = self._select_model(task.tier, task.estimated_tokens)
        if not model_id:
            raise RuntimeError(
                "No suitable model available — all models in relevant tiers are quota-exhausted."
            )

        self.last_used_model = model_id

        if on_chunk:
            response = self.groq_client.stream_complete(model_id, messages, on_chunk)
        else:
            response = self.groq_client.complete(model_id, messages)

        total_tokens = task.estimated_tokens + estimate_tokens(response)
        self.quota_tracker.record(model_id, total_tokens)

        # Handle tool calls
        tool_calls = parse_tool_calls(response)
        if tool_calls:
            response = self._handle_tool_calls_sync(
                tool_calls, messages, model_id, on_chunk
            )

        # Track in memory engine
        self.memory_engine.add_conversation_turn(prompt, response, model_id)
        return response

    # ─── Asynchronous Dispatch ─────────────────────────────────────────────────

    async def dispatch_async(
        self,
        prompt: str,
        on_chunk: Callable[[str], None] | None = None,
    ) -> str:
        """Asynchronously process a user prompt."""
        if not self._initialized:
            await self.initialize_async()

        task = self.task_router.classify(prompt)
        messages = self._build_messages(prompt)

        model_id = self._select_model(task.tier, task.estimated_tokens)
        if not model_id:
            raise RuntimeError(
                "No suitable model available — all models in relevant tiers are quota-exhausted."
            )

        self.last_used_model = model_id

        if on_chunk:
            response = await self.groq_client.stream_complete_async(
                model_id, messages, on_chunk
            )
        else:
            response = await self.groq_client.complete_async(model_id, messages)

        total_tokens = task.estimated_tokens + estimate_tokens(response)
        self.quota_tracker.record(model_id, total_tokens)

        # Handle tool calls
        tool_calls = parse_tool_calls(response)
        if tool_calls:
            response = await self._handle_tool_calls_async(
                tool_calls, messages, model_id, on_chunk
            )

        # Track in memory engine
        self.memory_engine.add_conversation_turn(prompt, response, model_id)
        return response

    # ─── Tool Call Handlers ────────────────────────────────────────────────────

    def _handle_tool_calls_sync(
        self,
        tool_calls: list[Any],
        base_messages: list[dict[str, str]],
        model_id: str,
        on_chunk: Callable[[str], None] | None = None,
    ) -> str:
        results = execute_tool_calls_sync(tool_calls, self.workspace_root)
        tool_summary = "\n\n".join(results)

        follow_up_messages = list(base_messages)
        follow_up_messages.append({
            "role": "user",
            "content": f"Tool execution results:\n\n{tool_summary}\n\nPlease summarize what was done and continue.",
        })

        if on_chunk:
            on_chunk("\n\n_Executing tools..._\n\n")
            follow_up = self.groq_client.stream_complete(
                model_id, follow_up_messages, on_chunk
            )
        else:
            follow_up = self.groq_client.complete(model_id, follow_up_messages)

        self.quota_tracker.record(model_id, estimate_tokens(follow_up))
        return follow_up

    async def _handle_tool_calls_async(
        self,
        tool_calls: list[Any],
        base_messages: list[dict[str, str]],
        model_id: str,
        on_chunk: Callable[[str], None] | None = None,
    ) -> str:
        results = await execute_tool_calls_async(tool_calls, self.workspace_root)
        tool_summary = "\n\n".join(results)

        follow_up_messages = list(base_messages)
        follow_up_messages.append({
            "role": "user",
            "content": f"Tool execution results:\n\n{tool_summary}\n\nPlease summarize what was done and continue.",
        })

        if on_chunk:
            on_chunk("\n\n_Executing tools..._\n\n")
            follow_up = await self.groq_client.stream_complete_async(
                model_id, follow_up_messages, on_chunk
            )
        else:
            follow_up = await self.groq_client.complete_async(
                model_id, follow_up_messages
            )

        self.quota_tracker.record(model_id, estimate_tokens(follow_up))
        return follow_up

    # ─── Model Selection & Message Assembly ───────────────────────────────────

    def _select_model(
        self, preferred_tier: TaskTier, estimated_tokens: int
    ) -> str | None:
        if self.selected_model and self.selected_model != "auto":
            model = self.model_registry.get_model(self.selected_model)
            if model and not model.disabled:
                if self.quota_tracker.can_accept(
                    model.id, model.limits, estimated_tokens
                ):
                    return model.id

        fallback_chain = TIER_FALLBACK_ORDER.get(
            preferred_tier, [preferred_tier, TaskTier.BALANCED, TaskTier.FAST]
        )

        statuses = self.get_quota_statuses()

        for tier in fallback_chain:
            candidates = self.model_registry.get_models_by_tier(tier)
            candidates.sort(
                key=lambda m: (
                    statuses.get(m.id).most_constrained.ratio
                    if statuses.get(m.id)
                    else 0.0
                )
            )

            for m in candidates:
                if self.quota_tracker.can_accept(m.id, m.limits, estimated_tokens):
                    return m.id

        return None

    def _build_messages(self, user_prompt: str) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []

        # System prompt + tool preamble
        extracted_sys = self.system_prompt_engine.extract_relevant(
            user_prompt, self.context_engine.read_context()
        )
        system_content = (
            f"{TOOL_PREAMBLE}\n\n{extracted_sys}" if extracted_sys else TOOL_PREAMBLE
        )
        messages.append({"role": "system", "content": system_content})

        # Workspace context
        workspace_ctx = self.context_engine.read_context()
        if workspace_ctx.strip():
            messages.append({
                "role": "system",
                "content": f"## Workspace Context (from context.md)\n{workspace_ctx}",
            })

        # Scored memory context
        memory_ctx = self.memory_engine.retrieve_relevant(user_prompt)
        if memory_ctx.strip():
            messages.append({
                "role": "system",
                "content": f"## Relevant Memory (from memory.md)\n{memory_ctx}",
            })

        # Active file context
        final_prompt = user_prompt
        active_str = self.active_file_tracker.get_active_context_string()
        if active_str:
            final_prompt = f"[{active_str}]\n\n{user_prompt}"

        messages.append({"role": "user", "content": final_prompt})
        return messages
