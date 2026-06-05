"""Bounded vision/OCR observation helpers for OpenRecall-adjacent consumers."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping
from uuid import uuid4

VERSION = "vision_observation_v1"
DEFAULT_PREVIEW_CHARS = 240


def _now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _collapse_ws(value: Any) -> str:
    return " ".join(str(value or "").split())


def _truncate(text: str, max_chars: int) -> str:
    collapsed = _collapse_ws(text)
    if len(collapsed) <= max_chars:
        return collapsed
    return collapsed[: max(1, max_chars) - 1].rstrip() + "…"


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _coerce_segments(segments: Iterable[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    coerced: list[dict[str, Any]] = []
    for segment in segments or []:
        if not isinstance(segment, Mapping):
            continue
        item = dict(segment)
        if "text" in item:
            item["text_hash"] = item.get("text_hash") or sha256_text(str(item.get("text") or ""))
            item["text_preview"] = item.get("text_preview") or _truncate(
                str(item.pop("text") or ""),
                DEFAULT_PREVIEW_CHARS,
            )
        coerced.append(item)
    return coerced


def build_vision_observation(
    *,
    image_ref: str | Path | None = None,
    image_hash: str | None = None,
    text: str = "",
    segments: Iterable[Mapping[str, Any]] | None = None,
    backend: str | None = None,
    elapsed_seconds: float | None = None,
    is_final: bool = True,
    confidence: float | None = None,
    source: str = "openrecall.vision",
    observation_id: str | None = None,
    ts: str | None = None,
    preview_chars: int = DEFAULT_PREVIEW_CHARS,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a non-authoritative observation over an image/OCR result.

    The returned payload intentionally stores bounded previews and hashes. Full
    text remains producer-owned unless a later explicit storage policy permits it.
    """

    image_ref_text = str(image_ref) if image_ref is not None else None
    resolved_image_hash = image_hash
    if resolved_image_hash is None and image_ref is not None and Path(image_ref).exists():
        resolved_image_hash = sha256_file(image_ref)

    clean_text = str(text or "")
    payload: dict[str, Any] = {
        "version": VERSION,
        "observation_id": observation_id or f"vision.obs:{uuid4()}",
        "ts": ts or _now_utc(),
        "source": source,
        "image_ref": image_ref_text,
        "image_hash": resolved_image_hash,
        "backend": backend,
        "elapsed_seconds": elapsed_seconds,
        "is_final": bool(is_final),
        "confidence": confidence,
        "text_preview": _truncate(clean_text, max(1, preview_chars)) if clean_text else None,
        "text_hash": sha256_text(clean_text) if clean_text else None,
        "segments": _coerce_segments(segments),
        "metadata": dict(metadata or {}),
        "pnf_candidates": [],
        "task_identity_residual": None,
        "lifecycle_residual": None,
        "kanban_projection_policy": "observer_only",
        "non_authoritative": True,
        "bounded": True,
    }
    payload["artifact_hash"] = sha256_text(json.dumps(payload, sort_keys=True, ensure_ascii=False))
    return payload


def observations_from_jsonl(path: str | Path) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid vision observation JSONL at line {line_no}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Vision observation JSONL line {line_no} must be an object")
        observations.append(normalize_observation_row(row))
    return observations


def normalize_observation_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a partial/final streaming row into the v1 observation contract."""

    payload = build_vision_observation(
        image_ref=row.get("image_ref") or row.get("screenshot_ref"),
        image_hash=row.get("image_hash"),
        text=str(row.get("text") or row.get("ocr_text") or ""),
        segments=row.get("segments") if isinstance(row.get("segments"), list) else None,
        backend=str(row.get("backend") or "") or None,
        elapsed_seconds=row.get("elapsed_seconds"),
        is_final=bool(row.get("is_final", True)),
        confidence=row.get("confidence"),
        source=str(row.get("source") or "openrecall.vision"),
        observation_id=row.get("observation_id"),
        ts=row.get("ts") or row.get("timestamp"),
        metadata=row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {},
    )
    if not payload.get("text_hash") and row.get("text_hash"):
        payload["text_hash"] = str(row.get("text_hash"))
    if not payload.get("text_preview") and row.get("text_preview"):
        payload["text_preview"] = str(row.get("text_preview"))
    if row.get("version") == VERSION and row.get("non_authoritative") is not None:
        payload["non_authoritative"] = bool(row.get("non_authoritative"))
    return payload


def write_observations_jsonl(path: str | Path, observations: Iterable[Mapping[str, Any]]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for observation in observations:
            handle.write(json.dumps(dict(observation), sort_keys=True) + "\n")
    return target
