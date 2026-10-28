"""Immutable, content-addressed snapshot events for screen/app capture.

The event is observer-class capture metadata. It does not promote OCR or window
metadata into semantic truth. Event identity is the SHA-256 digest of the
canonical event body. A chain links each event to the prior event id without
allowing in-place mutation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import Iterable

SCHEMA_VERSION = "openrecall.snapshot_event.v1"
CAPTURE_KIND = "screen_app"
HASH_ALGORITHM = "sha256"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_EVENT_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class SnapshotEventError(ValueError):
    """Raised when a snapshot event violates the v1 contract."""


@dataclass(frozen=True, slots=True)
class SnapshotEvent:
    schema_version: str
    event_id: str
    capture_kind: str
    captured_at_ms: int
    monitor_index: int
    app_name: str
    window_title: str
    screenshot_sha256: str
    screenshot_blob_ref: str
    ocr_sha256: str
    previous_event_id: str | None

    def to_dict(self) -> dict[str, object]:
        """Return the stable JSON-compatible event representation."""

        return asdict(self)


def snapshot_event_schema() -> dict[str, object]:
    """Return the deterministic JSON Schema for SnapshotEvent v1."""

    sha256 = "^[0-9a-f]{64}$"
    event_id = "^sha256:[0-9a-f]{64}$"
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SCHEMA_VERSION,
        "title": "OpenRecall screen/app snapshot event",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "event_id",
            "capture_kind",
            "captured_at_ms",
            "monitor_index",
            "app_name",
            "window_title",
            "screenshot_sha256",
            "screenshot_blob_ref",
            "ocr_sha256",
            "previous_event_id",
        ],
        "properties": {
            "schema_version": {"const": SCHEMA_VERSION},
            "event_id": {"type": "string", "pattern": event_id},
            "capture_kind": {"const": CAPTURE_KIND},
            "captured_at_ms": {"type": "integer", "minimum": 0},
            "monitor_index": {"type": "integer", "minimum": 0},
            "app_name": {"type": "string"},
            "window_title": {"type": "string"},
            "screenshot_sha256": {"type": "string", "pattern": sha256},
            "screenshot_blob_ref": {
                "type": "string",
                "pattern": "^sha256:[0-9a-f]{64}$",
            },
            "ocr_sha256": {"type": "string", "pattern": sha256},
            "previous_event_id": {
                "oneOf": [
                    {"type": "null"},
                    {"type": "string", "pattern": event_id},
                ]
            },
        },
    }


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json_bytes(value: dict[str, object]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _event_body(
    *,
    captured_at_ms: int,
    monitor_index: int,
    app_name: str,
    window_title: str,
    screenshot_sha256: str,
    ocr_sha256: str,
    previous_event_id: str | None,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "capture_kind": CAPTURE_KIND,
        "captured_at_ms": captured_at_ms,
        "monitor_index": monitor_index,
        "app_name": app_name,
        "window_title": window_title,
        "screenshot_sha256": screenshot_sha256,
        "screenshot_blob_ref": f"{HASH_ALGORITHM}:{screenshot_sha256}",
        "ocr_sha256": ocr_sha256,
        "previous_event_id": previous_event_id,
    }


def _event_id_for_body(body: dict[str, object]) -> str:
    return f"{HASH_ALGORITHM}:{_sha256_bytes(_canonical_json_bytes(body))}"


def build_snapshot_event(
    *,
    captured_at_ms: int,
    monitor_index: int,
    app_name: str,
    window_title: str,
    screenshot_bytes: bytes,
    ocr_text: str,
    previous_event_id: str | None = None,
) -> SnapshotEvent:
    """Build one immutable screen/app snapshot event."""

    if captured_at_ms < 0:
        raise SnapshotEventError("captured_at_ms must be non-negative")
    if monitor_index < 0:
        raise SnapshotEventError("monitor_index must be non-negative")
    if previous_event_id is not None and not _EVENT_ID_RE.fullmatch(previous_event_id):
        raise SnapshotEventError("previous_event_id must be a sha256 event id")

    screenshot_sha256 = _sha256_bytes(bytes(screenshot_bytes))
    ocr_sha256 = _sha256_bytes(ocr_text.encode("utf-8"))
    body = _event_body(
        captured_at_ms=captured_at_ms,
        monitor_index=monitor_index,
        app_name=str(app_name),
        window_title=str(window_title),
        screenshot_sha256=screenshot_sha256,
        ocr_sha256=ocr_sha256,
        previous_event_id=previous_event_id,
    )
    event = SnapshotEvent(event_id=_event_id_for_body(body), **body)
    validate_snapshot_event(event)
    return event


def validate_snapshot_event(event: SnapshotEvent) -> None:
    """Validate schema constants, content addresses, and event identity."""

    if event.schema_version != SCHEMA_VERSION:
        raise SnapshotEventError("unsupported schema_version")
    if event.capture_kind != CAPTURE_KIND:
        raise SnapshotEventError("capture_kind must be screen_app")
    if event.captured_at_ms < 0:
        raise SnapshotEventError("captured_at_ms must be non-negative")
    if event.monitor_index < 0:
        raise SnapshotEventError("monitor_index must be non-negative")
    if not _SHA256_RE.fullmatch(event.screenshot_sha256):
        raise SnapshotEventError("invalid screenshot_sha256")
    if event.screenshot_blob_ref != f"{HASH_ALGORITHM}:{event.screenshot_sha256}":
        raise SnapshotEventError("screenshot_blob_ref must address screenshot_sha256")
    if not _SHA256_RE.fullmatch(event.ocr_sha256):
        raise SnapshotEventError("invalid ocr_sha256")
    if event.previous_event_id is not None and not _EVENT_ID_RE.fullmatch(
        event.previous_event_id
    ):
        raise SnapshotEventError("invalid previous_event_id")
    if not _EVENT_ID_RE.fullmatch(event.event_id):
        raise SnapshotEventError("invalid event_id")

    body = event.to_dict()
    body.pop("event_id")
    expected = _event_id_for_body(body)
    if event.event_id != expected:
        raise SnapshotEventError("event_id does not match canonical event content")


def verify_append_only_chain(events: Iterable[SnapshotEvent]) -> None:
    """Verify event identity and strict predecessor linkage in order."""

    previous: SnapshotEvent | None = None
    seen: set[str] = set()

    for event in events:
        validate_snapshot_event(event)
        if event.event_id in seen:
            raise SnapshotEventError("duplicate event_id in append-only chain")

        expected_previous = previous.event_id if previous is not None else None
        if event.previous_event_id != expected_previous:
            raise SnapshotEventError("previous_event_id breaks append-only chain")

        seen.add(event.event_id)
        previous = event
