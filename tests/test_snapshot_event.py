from dataclasses import FrozenInstanceError
import hashlib

import pytest

from openrecall.snapshot_event import (
    CAPTURE_KIND,
    SCHEMA_VERSION,
    SnapshotEventError,
    build_snapshot_event,
    snapshot_event_schema,
    validate_snapshot_event,
    verify_append_only_chain,
)


def _event(previous_event_id=None, *, screenshot=b"screen-a", ocr_text="hello"):
    return build_snapshot_event(
        captured_at_ms=1_714_000_000_123,
        monitor_index=0,
        app_name="Browser",
        window_title="Matter review",
        screenshot_bytes=screenshot,
        ocr_text=ocr_text,
        previous_event_id=previous_event_id,
    )


def test_schema_is_closed_and_names_screen_app_capture():
    schema = snapshot_event_schema()

    assert schema["$id"] == SCHEMA_VERSION
    assert schema["additionalProperties"] is False
    assert schema["properties"]["capture_kind"]["const"] == CAPTURE_KIND
    assert "event_id" in schema["required"]
    assert "previous_event_id" in schema["required"]
    assert "screenshot_blob_ref" in schema["required"]


def test_snapshot_event_is_deterministically_content_addressed():
    left = _event()
    right = _event()

    assert left == right
    assert left.event_id.startswith("sha256:")
    assert left.screenshot_blob_ref == f"sha256:{left.screenshot_sha256}"
    assert left.screenshot_sha256 == hashlib.sha256(b"screen-a").hexdigest()
    assert left.ocr_sha256 == hashlib.sha256(b"hello").hexdigest()


def test_payload_change_changes_event_identity():
    baseline = _event()
    changed_screen = _event(screenshot=b"screen-b")
    changed_ocr = _event(ocr_text="different")

    assert changed_screen.event_id != baseline.event_id
    assert changed_ocr.event_id != baseline.event_id


def test_event_is_immutable():
    event = _event()

    with pytest.raises(FrozenInstanceError):
        event.app_name = "Mail"


def test_append_only_chain_accepts_exact_predecessors():
    first = _event()
    second = _event(first.event_id, screenshot=b"screen-b")
    third = _event(second.event_id, screenshot=b"screen-c")

    verify_append_only_chain([first, second, third])


def test_append_only_chain_rejects_missing_predecessor():
    first = _event()
    second = _event(first.event_id, screenshot=b"screen-b")

    with pytest.raises(SnapshotEventError, match="previous_event_id"):
        verify_append_only_chain([second, first])


def test_append_only_chain_rejects_duplicate_event():
    first = _event()

    with pytest.raises(SnapshotEventError, match="duplicate event_id"):
        verify_append_only_chain([first, first])


def test_validate_rejects_event_with_tampered_content_address():
    event = _event()
    tampered = object.__new__(type(event))

    for field_name in event.__dataclass_fields__:
        object.__setattr__(tampered, field_name, getattr(event, field_name))
    object.__setattr__(tampered, "window_title", "tampered")

    with pytest.raises(SnapshotEventError, match="event_id does not match"):
        validate_snapshot_event(tampered)


@pytest.mark.parametrize(
    ("captured_at_ms", "monitor_index"),
    [(-1, 0), (0, -1)],
)
def test_builder_rejects_negative_coordinates(captured_at_ms, monitor_index):
    with pytest.raises(SnapshotEventError):
        build_snapshot_event(
            captured_at_ms=captured_at_ms,
            monitor_index=monitor_index,
            app_name="Browser",
            window_title="Matter review",
            screenshot_bytes=b"screen",
            ocr_text="text",
        )


def test_builder_rejects_invalid_previous_event_id():
    with pytest.raises(SnapshotEventError, match="previous_event_id"):
        build_snapshot_event(
            captured_at_ms=0,
            monitor_index=0,
            app_name="Browser",
            window_title="Matter review",
            screenshot_bytes=b"screen",
            ocr_text="text",
            previous_event_id="not-a-content-address",
        )
