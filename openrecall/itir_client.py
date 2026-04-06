from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


class ItirBridgeError(RuntimeError):
    pass


def _suite_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _bridge_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_suite_root() / "itir-mcp" / "src")
    return env


def _run_bridge(request: Mapping[str, Any], *, timeout_seconds: float = 30.0) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-m", "itir_mcp", "--bridge"],
        cwd=_suite_root() / "itir-mcp",
        env=_bridge_env(),
        input=json.dumps(dict(request), sort_keys=True) + "\n",
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_seconds,
    )
    if completed.returncode != 0:
        raise ItirBridgeError(
            f"itir-mcp bridge failed with exit code {completed.returncode}: {(completed.stderr or '').strip()}"
        )
    line = (completed.stdout or "").strip().splitlines()
    if not line:
        raise ItirBridgeError("itir-mcp bridge returned no output")
    try:
        response = json.loads(line[0])
    except json.JSONDecodeError as exc:
        raise ItirBridgeError(f"itir-mcp bridge returned invalid json: {line[0]}") from exc
    if not isinstance(response, dict):
        raise ItirBridgeError("itir-mcp bridge returned a non-object response")
    if response.get("ok") is False:
        error = response.get("error") if isinstance(response.get("error"), dict) else {}
        raise ItirBridgeError(
            f"{error.get('code', 'tool_error')}: {error.get('message', 'unknown error')}"
        )
    result = response.get("result")
    if not isinstance(result, dict):
        raise ItirBridgeError("itir-mcp bridge returned a non-object result payload")
    return result


def call_tool(name: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    return _run_bridge({"op": "call", "name": name, "payload": dict(payload)})


def safe_call_tool(name: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    return _run_bridge({"op": "safe_call", "name": name, "payload": dict(payload)})


def _verified_tool_result(name: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    guarded = safe_call_tool(name, payload)
    decision = str(guarded.get("decision") or "")
    result = guarded.get("result")
    if decision != "verified" or not isinstance(result, dict):
        explanation = guarded.get("status_explanation") if isinstance(guarded.get("status_explanation"), dict) else {}
        reason_codes = explanation.get("reason_codes") if isinstance(explanation.get("reason_codes"), list) else []
        reasons = ", ".join(str(code) for code in reason_codes if str(code).strip()) or str(explanation.get("primary_reason_code") or decision or "guarded_failure")
        raise ItirBridgeError(f"itir guarded tool call did not verify: {reasons}")
    return result


def compare_observations(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    return _verified_tool_result("itir.compare_observations", {"left": dict(left), "right": dict(right)})


def score_coherence(observations: list[Mapping[str, Any]]) -> dict[str, Any]:
    return _verified_tool_result("itir.score_coherence", {"observations": [dict(row) for row in observations]})


def build_envelope(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    comparison: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"left": dict(left), "right": dict(right)}
    if comparison is not None:
        payload["comparison"] = dict(comparison)
    return _verified_tool_result("itir.build_envelope", payload)
