# Copyright (c) 2026 Frank1o3
# SPDX-License-Identifier: MIT

"""GroqClient — Async and Sync wrapper around the official groq SDK.

Provides complete() and stream_complete() methods,
for both synchronous and asynchronous operations.
"""

import os
from typing import TYPE_CHECKING, cast

from groq import AsyncGroq, Groq
from groq.types.chat import ChatCompletionMessageParam

if TYPE_CHECKING:
    from collections.abc import Callable

    from .types import OrchestratorConfig


class GroqClient:
    """Wrapper around Groq API for chat completion and streaming."""

    def __init__(self, config: OrchestratorConfig) -> None:
        self.config = config
        api_key = config.api_key or os.environ.get("GROQ_API_KEY") or None
        base_url = config.base_url or None

        self._async_client = AsyncGroq(api_key=api_key, base_url=base_url)
        self._sync_client = Groq(api_key=api_key, base_url=base_url)

    @property
    def raw_async(self) -> AsyncGroq:
        """Access raw AsyncGroq client."""
        return self._async_client

    @property
    def raw_sync(self) -> Groq:
        """Access raw sync Groq client."""
        return self._sync_client

    @staticmethod
    def _sdk_messages(
        messages: list[dict[str, str]],
    ) -> list[ChatCompletionMessageParam]:
        """Convert the engine's simple message representation for the SDK boundary."""
        return cast(list[ChatCompletionMessageParam], messages)

    # ─── Async API ─────────────────────────────────────────────────────────────

    async def complete_async(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> str:
        """Send asynchronous chat completion request and return response text."""
        response = await self._async_client.chat.completions.create(
            model=model,
            messages=self._sdk_messages(messages),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return str(response.choices[0].message.content or "")

    async def stream_complete_async(
        self,
        model: str,
        messages: list[dict[str, str]],
        on_chunk: Callable[[str], None] | None = None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> str:
        """Stream asynchronous completion, invoking on_chunk for deltas, resolving with full text."""
        stream = await self._async_client.chat.completions.create(
            model=model,
            messages=self._sdk_messages(messages),
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        full_text: list[str] = []
        async for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                full_text.append(delta)
                if on_chunk:
                    on_chunk(delta)

        return "".join(full_text)

    # ─── Sync API ──────────────────────────────────────────────────────────────

    def complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> str:
        """Send synchronous chat completion request and return response text."""
        response = self._sync_client.chat.completions.create(
            model=model,
            messages=self._sdk_messages(messages),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return str(response.choices[0].message.content or "")

    def stream_complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        on_chunk: Callable[[str], None] | None = None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> str:
        """Stream synchronous completion, invoking on_chunk for deltas, returning full text."""
        stream = self._sync_client.chat.completions.create(
            model=model,
            messages=self._sdk_messages(messages),
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        full_text: list[str] = []
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                full_text.append(delta)
                if on_chunk:
                    on_chunk(delta)

        return "".join(full_text)
