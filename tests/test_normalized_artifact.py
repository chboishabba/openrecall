import json
import sys
from pathlib import Path

import jsonschema


ROOT = Path("/home/c/Documents/code/ITIR-suite")
OPENRECALL_ROOT = ROOT / "openrecall"
SCHEMA_PATH = ROOT / "schemas" / "itir.normalized.artifact.v1.schema.json"

sys.path.insert(0, str(OPENRECALL_ROOT / "openrecall"))

from normalized_artifact import (
    build_normalized_sidecar,
    build_search_follow_artifact,
    build_search_follow_normalized_artifact,
    write_sidecar,
)


def _validator() -> jsonschema.Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return jsonschema.Draft202012Validator(schema)


def test_openrecall_normalized_sidecar_validates(tmp_path: Path) -> None:
    capture_path = tmp_path / "captures" / "20260402_120000.webp"
    capture_path.parent.mkdir(parents=True, exist_ok=True)
    capture_path.write_bytes(b"capture")

    payload = build_normalized_sidecar(
        artifact_id="openrecall.capture:20260402_120000",
        capture_path=capture_path,
        aliases=["openrecall:latest"],
    )

    _validator().validate(payload)
    assert payload["artifact_role"] == "source_artifact"
    assert payload["authority"]["authority_class"] == "capture"
    assert payload["authority"]["derived"] is False
    assert payload["canonical_identity"]["identity_key"] == capture_path.name
    assert payload["lineage"]["upstream_artifact_ids"] == [str(capture_path)]
    assert payload["unresolved_pressure_status"] == "none"


def test_openrecall_sidecar_writer_round_trips(tmp_path: Path) -> None:
    payload = build_normalized_sidecar(
        artifact_id="openrecall.capture:test-1",
        capture_path=tmp_path / "capture.webp",
    )
    output_path = tmp_path / "capture.normalized.json"
    write_sidecar(payload, output_path)

    assert output_path.exists()
    reloaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert reloaded == payload


def test_openrecall_search_follow_artifact_is_bounded_and_non_authoritative() -> None:
    results = [
        {
            "id": 1,
            "app": "Browser",
            "title": "Brexit note",
            "text": "Brexit legal interaction note",
            "timestamp": 1712000000,
        },
        {
            "id": 2,
            "app": "Mail",
            "title": "EU follow-up",
            "text": "Need to review the bounded result set carefully.",
            "timestamp": 1712000001,
        },
    ]

    artifact = build_search_follow_artifact(
        "brexit legal interaction",
        results,
        max_results=1,
        stop_after=True,
    )

    assert artifact["artifact_type"] == "derived.openrecall.search_follow"
    assert artifact["non_authoritative"] is True
    assert artifact["bounded"] is True
    assert artifact["trigger"]["command"] == "openrecall search"
    assert artifact["scope"]["max_results"] == 1
    assert len(artifact["results"]) == 1


def test_openrecall_search_follow_normalized_artifact_validates() -> None:
    follow_artifact = build_search_follow_artifact(
        "privacy memory",
        [
            {
                "id": 7,
                "app": "Notes",
                "title": "Privacy memo",
                "text": "Recall stays local-first and bounded.",
                "timestamp": 1712000100,
            }
        ],
    )

    payload = build_search_follow_normalized_artifact(follow_artifact)

    _validator().validate(payload)
    assert payload["artifact_role"] == "derived_product"
    assert payload["authority"]["authority_class"] == "derived_inspection"
    assert payload["authority"]["derived"] is True
    assert payload["unresolved_pressure_status"] == "follow_needed"
    assert payload["follow_obligation"]["trigger"] == "privacy memory"
    assert payload["summary"]["result_count"] == 1
