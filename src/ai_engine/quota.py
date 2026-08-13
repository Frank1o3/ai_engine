# Copyright (c) 2026 Frank1o3
# SPDX-License-Identifier: MIT

"""QuotaTracker — sliding-window usage tracking per model.

Logs timestamped entries with token counts. Expired entries auto-prune on checks,
providing exact RPM/TPM (60s) and RPD/TPD (24h) status.
"""

import time
from dataclasses import dataclass

from .types import (
    ConstrainedDimension,
    ModelLimits,
    ModelQuotaStatus,
    QuotaHealth,
    QuotaUsage,
)

MINUTE_SECONDS = 60.0
DAY_SECONDS = 24.0 * 60.0 * 60.0


@dataclass(slots=True)
class UsageEntry:
    """Timestamped request log entry."""

    timestamp: float  # Unix timestamp in seconds
    tokens: int  # Token count for the request


class QuotaTracker:
    """Sliding-window quota tracker across RPM/RPD/TPM/TPD."""

    def __init__(self, warning_threshold: float = 0.9) -> None:
        self.warning_threshold = warning_threshold
        self._logs: dict[str, list[UsageEntry]] = {}

    def record(self, model_id: str, tokens: int) -> None:
        """Record request timestamp and token count."""
        entries = self._logs.setdefault(model_id, [])
        entries.append(UsageEntry(timestamp=time.time(), tokens=tokens))

    def _prune(self, model_id: str, window_seconds: float) -> None:
        """Prune log entries older than window_seconds."""
        entries = self._logs.get(model_id)
        if not entries:
            return
        cutoff = time.time() - window_seconds
        filtered = [e for e in entries if e.timestamp > cutoff]
        if len(filtered) != len(entries):
            self._logs[model_id] = filtered

    def get_usage(self, model_id: str) -> QuotaUsage:
        """Compute current usage metrics for a model across sliding windows."""
        self._prune(model_id, DAY_SECONDS)
        entries = self._logs.get(model_id, [])

        minute_cutoff = time.time() - MINUTE_SECONDS
        rpm_count = 0
        tpm_count = 0

        for entry in entries:
            if entry.timestamp > minute_cutoff:
                rpm_count += 1
                tpm_count += entry.tokens

        rpd_count = len(entries)
        tpd_count = sum(e.tokens for e in entries)

        return QuotaUsage(
            rpm_count=rpm_count,
            rpd_count=rpd_count,
            tpm_count=tpm_count,
            tpd_count=tpd_count,
        )

    def can_accept(
        self, model_id: str, limits: ModelLimits, estimated_tokens: int
    ) -> bool:
        """Check if a new request would exceed limits for the given model."""
        usage = self.get_usage(model_id)

        projected_rpm = usage.rpm_count + 1
        projected_rpd = usage.rpd_count + 1
        projected_tpm = usage.tpm_count + estimated_tokens
        projected_tpd = usage.tpd_count + estimated_tokens

        return (
            projected_rpm <= limits.rpm
            and projected_rpd <= limits.rpd
            and projected_tpm <= limits.tpm
            and projected_tpd <= limits.tpd
        )

    def get_health(self, model_id: str, limits: ModelLimits) -> ModelQuotaStatus:
        """Compute model health and identify the most constrained limit dimension."""
        usage = self.get_usage(model_id)

        ratios = {
            "rpm": usage.rpm_count / (limits.rpm or 1.0),
            "rpd": usage.rpd_count / (limits.rpd or 1.0),
            "tpm": usage.tpm_count / (limits.tpm or 1.0),
            "tpd": usage.tpd_count / (limits.tpd or 1.0),
        }

        max_key = "rpm"
        max_ratio = 0.0
        for key, ratio in ratios.items():
            if ratio > max_ratio:
                max_ratio = ratio
                max_key = key

        if max_ratio >= 1.0:
            health = QuotaHealth.EXHAUSTED
        elif max_ratio >= self.warning_threshold:
            health = QuotaHealth.WARNING
        else:
            health = QuotaHealth.HEALTHY

        return ModelQuotaStatus(
            model_id=model_id,
            health=health,
            limits=limits,
            usage=usage,
            most_constrained=ConstrainedDimension(key=max_key, ratio=max_ratio),
        )

    def get_all_health(
        self, model_limits: dict[str, ModelLimits]
    ) -> dict[str, ModelQuotaStatus]:
        """Compute health status for all registered models."""
        return {
            m_id: self.get_health(m_id, limits) for m_id, limits in model_limits.items()
        }

    def reset(self, model_id: str) -> None:
        """Reset logs for a single model."""
        self._logs.pop(model_id, None)

    def reset_all(self) -> None:
        """Reset all logs."""
        self._logs.clear()

    def set_warning_threshold(self, threshold: float) -> None:
        """Update runtime warning threshold ratio (e.g. 0.9)."""
        self.warning_threshold = threshold
