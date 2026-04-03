import json
import sys
from pathlib import Path

import jsonschema


ROOT = Path("/home/c/Documents/code/ITIR-suite")
OPENRECALL_ROOT = ROOT / "openrecall"
SCHEMA_PATH = ROOT / "schemas" / "itir.normalized.artifact.v1.schema.json"

sys.path.insert(0, str(OPENRECALL_ROOT / "openrecall"))

from retrieval_follow import (  # noqa: E402
    build_retrieval_follow_normalized_artifact,
    build_retrieval_follow_sidecar,
)


def _validator() -> jsonschema.Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return jsonschema.Draft202012Validator(schema)


def test_build_retrieval_follow_sidecar_truncates_results() -> None:
    results = [
        {"id": 1, "app": "Browser", "title": "Request", "text": "request 42", "timestamp": 1712000000},
        {"id": 2, "app": "Mail", "title": "Follow-up", "text": "follow up", "timestamp": 1712000001},
        {"id": 3, "app": "Notes", "title": "Extra", "text": "extra", "timestamp": 1712000002},
    ]

    sidecar = build_retrieval_follow_sidecar(
        "request 42",
        results,
        max_results=2,
        stop_after=False,
    )

    assert sidecar["artifact_type"] == "derived.openrecall.search_follow"
    assert sidecar["non_authoritative"] is True
    assert sidecar["bounded"] is True
    assert sidecar["scope"]["max_results"] == 2
    assert sidecar["scope"]["stop_after"] is False
    assert len(sidecar["results"]) == 2


def test_build_retrieval_follow_normalized_artifact_validates() -> None:
    results = [
        {"id": 7, "app": "Notes", "title": "Privacy", "text": "privacy memory", "timestamp": 1712000100},
    ]

    payload = build_retrieval_follow_normalized_artifact(
        "privacy memory",
        results,
    )

    _validator().validate(payload)
    assert payload["artifact_role"] == "derived_product"
    assert payload["authority"]["authority_class"] == "derived_inspection"
    assert payload["authority"]["derived"] is True
    assert payload["unresolved_pressure_status"] == "follow_needed"
    assert payload["follow_obligation"]["trigger"] == "privacy memory"
    assert payload["summary"]["result_count"] == 1
