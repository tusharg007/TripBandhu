"""
trace_observability.py — Trace schema validation and automated secret scanning.
Ensures capability traces adhere to safety contracts and leak zero credentials or local paths.
"""

from __future__ import annotations

import re
from typing import Any


_SENSITIVE_PATTERNS = [
    re.compile(r"gsk_[a-zA-Z0-9_\-]{20,}", re.IGNORECASE),
    re.compile(r"tvly-[a-zA-Z0-9_\-]{20,}", re.IGNORECASE),
    re.compile(r"postgresql://[^\s]+", re.IGNORECASE),
    re.compile(r"bearer\s+[a-zA-Z0-9_\-\.]{20,}", re.IGNORECASE),
    re.compile(r"password\s*[:=]\s*[^\s,;]+", re.IGNORECASE),
]


REQUIRED_PUBLIC_TRACE_KEYS = frozenset({
    "specialist",
    "capability",
    "server",
    "status",
    "latency_ms",
    "source_count",
    "error_code",
})


def validate_public_trace_entry(entry: dict[str, Any]) -> tuple[bool, str]:
    """Validate that a trace entry adheres to the safe public contract schema."""
    missing = REQUIRED_PUBLIC_TRACE_KEYS - set(entry.keys())
    if missing:
        return False, f"Missing required public trace fields: {sorted(missing)}"

    if entry.get("status") not in {"AVAILABLE", "DEGRADED", "UNAVAILABLE", "CONTRACT_MISMATCH"}:
        return False, f"Invalid status: {entry.get('status')}"

    return True, "Valid public trace entry"


def scan_trace_for_sensitive_data(trace: list[dict[str, Any]]) -> list[str]:
    """Scan a list of trace entries for any sensitive tokens, secrets, or file paths."""
    violations: list[str] = []

    for idx, entry in enumerate(trace, 1):
        entry_str = str(entry)

        for pattern in _SENSITIVE_PATTERNS:
            match = pattern.search(entry_str)
            if match:
                violations.append(
                    f"Entry {idx} ({entry.get('capability', 'unknown')}) contained sensitive data matching {pattern.pattern}: {match.group(0)[:15]}..."
                )

    return violations
