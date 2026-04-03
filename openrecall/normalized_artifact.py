"""Support for emitting `itir.normalized.artifact.v1` sidecars from OpenRecall captures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

SCHEMA_VERSION = "itir.normalized.artifact.v1"
CAPTURE_AUTHORITY_CLASS = "capture"


def build_normalized_sidecar(
    artifact_id: str,
    capture_path: Path | str,
    anchor_kind: str = "openrecall.capture",
    envelope_kind: str = "openrecall.envelope",
    identity_class: str = "openrecall_capture",
    authority_class: str = CAPTURE_AUTHORITY_CLASS,
    unresolved_status: str = "none",
    aliases: Iterable[str] | None = None,
) -> dict[str, object]:
    capture_path = Path(capture_path)
    canonical_identity: dict[str, object] = {
        "identity_class": identity_class,
        "identity_key": capture_path.name,
    }
    if aliases:
        canonical_identity["aliases"] = list(aliases)

    sidecar: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_role": "source_artifact",
        "artifact_id": artifact_id,
        "canonical_identity": canonical_identity,
        "provenance_anchor": {
            "source_system": "openrecall",
            "source_artifact_id": artifact_id,
            "anchor_kind": anchor_kind,
            "anchor_ref": str(capture_path),
        },
        "context_envelope_ref": {
            "envelope_id": f"{artifact_id}-envelope",
            "envelope_kind": envelope_kind,
        },
        "authority": {
            "authority_class": authority_class,
            "derived": False,
        },
        "lineage": {
            "upstream_artifact_ids": [str(capture_path)],
        },
        "unresolved_pressure_status": unresolved_status,
    }
    return sidecar


def write_sidecar(sidecar: dict[str, object], output_path: Path, *, indent: int = 2) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sidecar, indent=indent, sort_keys=True) + "\n", encoding="utf-8")


def _coerce_search_result(row: Any, index: int) -> dict[str, object]:
    getter = getattr(row, "_asdict", None)
    raw = getter() if callable(getter) else row
    if not isinstance(raw, dict):
        raw = {
            "id": getattr(row, "id", None),
            "app": getattr(row, "app", None),
            "title": getattr(row, "title", None),
            "text": getattr(row, "text", None),
            "timestamp": getattr(row, "timestamp", None),
        }

    text = str(raw.get("text") or "")
    return {
        "entry_id": str(raw.get("id") or f"entry:{index}"),
        "app": str(raw.get("app") or ""),
        "title": str(raw.get("title") or ""),
        "timestamp": raw.get("timestamp"),
        "text_excerpt": text[:240],
        "screenshot_ref": str(raw.get("timestamp")) + ".webp" if raw.get("timestamp") is not None else None,
    }


def build_search_follow_artifact(
    query: str,
    results: Iterable[Any],
    *,
    max_results: int = 5,
    stop_after: bool = True,
) -> dict[str, object]:
    limited_results = [
        _coerce_search_result(row, index)
        for index, row in enumerate(list(results)[:max_results], start=1)
    ]

    return {
        "artifact_id": f"derived.follow.openrecall.search.{uuid4()}",
        "artifact_type": "derived.openrecall.search_follow",
        "query": query,
        "results": limited_results,
        "non_authoritative": True,
        "bounded": True,
        "trigger": {
            "command": "openrecall search",
            "params": {"max_results": max_results, "stop_after": stop_after},
        },
        "scope": {
            "max_results": max_results,
            "stop_after": stop_after,
        },
    }


def build_search_follow_normalized_artifact(
    search_follow: dict[str, Any],
    *,
    profile_version: str = "openrecall.search_follow.v1",
) -> dict[str, object]:
    artifact_id = str(search_follow.get("artifact_id") or f"derived.follow.openrecall.search.{uuid4()}")
    query = str(search_follow.get("query") or "").strip()
    results = [row for row in (search_follow.get("results") or []) if isinstance(row, dict)]
    stop_after = bool((search_follow.get("scope") or {}).get("stop_after", True))
    max_results = int((search_follow.get("scope") or {}).get("max_results", len(results) or 1))

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_role": "derived_product",
        "artifact_id": f"openrecall.search_follow:{artifact_id}",
        "canonical_identity": {
            "identity_class": "openrecall_search_follow",
            "identity_key": artifact_id,
            "aliases": [f"query:{query}"] if query else [],
        },
        "provenance_anchor": {
            "source_system": "openrecall",
            "source_artifact_id": artifact_id,
            "anchor_kind": "search_follow_result",
            "anchor_ref": query or None,
        },
        "context_envelope_ref": {
            "envelope_id": f"openrecall.search:{query or 'unknown'}",
            "envelope_kind": "openrecall_search_context",
        },
        "authority": {
            "authority_class": "derived_inspection",
            "derived": True,
            "promotion_receipt_ref": None,
        },
        "lineage": {
            "upstream_artifact_ids": [
                str(row.get("screenshot_ref") or row.get("entry_id") or f"result:{index}")
                for index, row in enumerate(results, start=1)
            ],
            "profile_version": profile_version,
        },
        "follow_obligation": {
            "trigger": query or "openrecall_search_follow",
            "scope": f"review up to {max_results} OpenRecall search results",
            "stop_condition": (
                "stop after reviewing the bounded result set"
                if stop_after
                else "continue only under an explicit new bounded follow decision"
            ),
        },
        "unresolved_pressure_status": "follow_needed",
        "summary": {
            "producer": "openrecall",
            "query": query,
            "result_count": len(results),
            "non_authoritative": bool(search_follow.get("non_authoritative", True)),
            "bounded": bool(search_follow.get("bounded", True)),
            "stop_after": stop_after,
        },
    }
