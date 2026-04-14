"""Thin OpenRecall client for the canonical ITIR MCP comparison seam.

The current implementation uses the local `itir-mcp` JSON bridge transport, but
the contract remains MCP-first and tool-name stable.
"""

from __future__ import annotations

from .itir_client import (
    ItirBridgeError as ItirMcpError,
    build_envelope,
    call_tool,
    compare_observations,
    compare_observations_with_policy_hint,
    safe_call_tool,
    score_coherence,
)

__all__ = [
    "ItirMcpError",
    "build_envelope",
    "call_tool",
    "compare_observations",
    "compare_observations_with_policy_hint",
    "safe_call_tool",
    "score_coherence",
]
