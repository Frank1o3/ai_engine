# Copyright (c) 2026 Frank1o3
# SPDX-License-Identifier: MIT

"""ModelRegistry — fetches available models from Groq SDK,
enriches them with tier classification and quota limits,
and maintains the active model registry.
"""

import logging

from groq import AsyncGroq, Groq

from .types import (
    ModelLimits,
    RegisteredModel,
    TaskTier,
    resolve_tier,
)

logger = logging.getLogger(__name__)

# Default per-model limits (approximations based on public Groq documentation)
# Default per-model limits (approximations based on public Groq documentation)
DEFAULT_MODEL_LIMITS: dict[str, ModelLimits] = {
    "allam-2-7b": ModelLimits(rpm=30, rpd=7000, tpm=6000, tpd=500000),
    "groq/compound": ModelLimits(
        rpm=30,
        rpd=250,
        tpm=70000,
        tpd=float("inf"),
    ),
    "groq/compound-mini": ModelLimits(
        rpm=30,
        rpd=250,
        tpm=70000,
        tpd=float("inf"),
    ),
    "llama-3.1-8b-instant": ModelLimits(
        rpm=30,
        rpd=14400,
        tpm=6000,
        tpd=500000,
    ),
    "llama-3.3-70b-versatile": ModelLimits(
        rpm=30,
        rpd=1000,
        tpm=12000,
        tpd=100000,
    ),
    "openai/gpt-oss-120b": ModelLimits(
        rpm=30,
        rpd=1000,
        tpm=8000,
        tpd=200000,
    ),
    "openai/gpt-oss-20b": ModelLimits(
        rpm=30,
        rpd=1000,
        tpm=8000,
        tpd=200000,
    ),
    "qwen/qwen3.6-27b": ModelLimits(
        rpm=30,
        rpd=1000,
        tpm=8000,
        tpd=200000,
    ),
}

FALLBACK_LIMITS = ModelLimits(rpm=5000, rpd=100000, tpm=40000, tpd=100000)


class ModelRegistry:
    """Registry managing available LLM models, tier mappings, and limits."""

    def __init__(self, client: Groq | AsyncGroq | None = None) -> None:
        self.client = client
        self._models: list[RegisteredModel] = []
        self._limit_overrides: dict[str, ModelLimits] = {}

    def set_limits(self, model_id: str, limits: ModelLimits) -> None:
        """Override limits for a specific model."""
        self._limit_overrides[model_id] = limits

    def refresh(self) -> list[RegisteredModel]:
        """Fetch models from Groq API synchronously and build registered model list."""
        if isinstance(self.client, AsyncGroq):
            # Fallback to offline defaults if called synchronously on async client
            logger.warning(
                "[ModelRegistry] Synchronous refresh called on AsyncGroq client; using fallback."
            )
            self._models = self._get_fallback_models()
            return self._models

        if self.client is None:
            self._models = self._get_fallback_models()
            return self._models

        try:
            response = self.client.models.list()
            raw_models = getattr(response, "data", []) or []

            filtered_models: list[RegisteredModel] = []
            for m in raw_models:
                m_id = getattr(m, "id", None)
                if not m_id or m_id.startswith("whisper"):
                    continue

                tier = resolve_tier(m_id)
                limits = self._limit_overrides.get(
                    m_id, DEFAULT_MODEL_LIMITS.get(m_id, FALLBACK_LIMITS)
                )

                filtered_models.append(
                    RegisteredModel(
                        id=m_id,
                        name=m_id,
                        tier=tier,
                        limits=limits,
                        disabled=False,
                    )
                )

            self._models = filtered_models or self._get_fallback_models()
            return self._models
        except Exception as err:
            logger.warning(
                "[ModelRegistry] Failed to fetch models from API: %s. Using fallback list.",
                err,
            )
            self._models = self._get_fallback_models()
            return self._models

    async def refresh_async(self) -> list[RegisteredModel]:
        """Fetch models from Groq API asynchronously."""
        if not isinstance(self.client, AsyncGroq):
            return self.refresh()

        try:
            response = await self.client.models.list()
            raw_models = getattr(response, "data", []) or []

            filtered_models: list[RegisteredModel] = []
            for m in raw_models:
                m_id = getattr(m, "id", None)
                if not m_id or m_id.startswith("whisper"):
                    continue

                tier = resolve_tier(m_id)
                limits = self._limit_overrides.get(
                    m_id, DEFAULT_MODEL_LIMITS.get(m_id, FALLBACK_LIMITS)
                )

                filtered_models.append(
                    RegisteredModel(
                        id=m_id,
                        name=m_id,
                        tier=tier,
                        limits=limits,
                        disabled=False,
                    )
                )

            self._models = filtered_models or self._get_fallback_models()
            return self._models
        except Exception as err:
            logger.warning(
                "[ModelRegistry] Async API fetch failed: %s. Using fallback list.", err
            )
            self._models = self._get_fallback_models()
            return self._models

    def get_models(self) -> list[RegisteredModel]:
        """Return cached list of registered models."""
        if not self._models:
            self._models = self._get_fallback_models()
        return self._models

    def get_models_by_tier(self, tier: TaskTier) -> list[RegisteredModel]:
        """Get models for a specific tier, ignoring disabled ones."""
        return [m for m in self.get_models() if m.tier == tier and not m.disabled]

    def get_model(self, model_id: str) -> RegisteredModel | None:
        """Lookup single model by ID."""
        for m in self.get_models():
            if m.id == model_id:
                return m
        return None

    def disable_model(self, model_id: str) -> None:
        """Disable a model."""
        m = self.get_model(model_id)
        if m:
            m.disabled = True

    def enable_model(self, model_id: str) -> None:
        """Enable a model."""
        m = self.get_model(model_id)
        if m:
            m.disabled = False

    def _get_fallback_models(self) -> list[RegisteredModel]:
        """Build fallback models when API is unavailable."""
        return [
            RegisteredModel(
                id=m_id,
                name=m_id,
                tier=resolve_tier(m_id),
                limits=self._limit_overrides.get(
                    m_id, DEFAULT_MODEL_LIMITS.get(m_id, FALLBACK_LIMITS)
                ),
                disabled=False,
            )
            for m_id in DEFAULT_MODEL_LIMITS
        ]
