import json
import sys
from pathlib import Path

ROOT = Path("/home/c/Documents/code/ITIR-suite")
OPENRECALL_ROOT = ROOT / "openrecall"
sys.path.insert(0, str(OPENRECALL_ROOT / "openrecall"))

from vision_observation import (
    VERSION,
    build_vision_observation,
    observations_from_jsonl,
    sha256_file,
    write_observations_jsonl,
)


def test_build_vision_observation_is_bounded_and_non_authoritative(tmp_path: Path) -> None:
    image_path = tmp_path / "capture.png"
    image_path.write_bytes(b"fake image")

    payload = build_vision_observation(
        image_ref=image_path,
        text="alpha beta gamma delta",
        segments=[{"text": "alpha beta", "box": [1, 2, 3, 4]}],
        backend="playwright_screenshot_ocr",
        preview_chars=12,
        confidence=0.8,
    )

    assert payload["version"] == VERSION
    assert payload["image_hash"] == sha256_file(image_path)
    assert payload["text_preview"] == "alpha beta…"
    assert payload["text_hash"].startswith("sha256:")
    assert payload["non_authoritative"] is True
    assert payload["bounded"] is True
    assert payload["pnf_candidates"] == []
    assert payload["kanban_projection_policy"] == "observer_only"
    assert payload["task_identity_residual"] is None
    assert payload["lifecycle_residual"] is None
    assert payload["artifact_hash"].startswith("sha256:")
    assert payload["segments"][0]["text_preview"] == "alpha beta"
    assert payload["segments"][0]["text_hash"].startswith("sha256:")
    assert "text" not in payload["segments"][0]
    assert "task_id" not in payload
    assert "kanban_card_id" not in payload
    assert "workflow_state" not in payload


def test_observations_from_jsonl_normalizes_partial_and_final_rows(tmp_path: Path) -> None:
    path = tmp_path / "vision.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "ts": "2026-06-04T00:00:00Z",
                        "source": "playwright_browser_assist",
                        "screenshot_ref": "shot-1.png",
                        "ocr_text": "partial page text",
                        "is_final": False,
                    }
                ),
                json.dumps(
                    {
                        "ts": "2026-06-04T00:00:01Z",
                        "image_ref": "shot-1.png",
                        "text": "final page text",
                        "is_final": True,
                        "metadata": {"latency_ms": 42},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = observations_from_jsonl(path)

    assert len(rows) == 2
    assert rows[0]["is_final"] is False
    assert rows[0]["image_ref"] == "shot-1.png"
    assert rows[1]["is_final"] is True
    assert rows[1]["metadata"] == {"latency_ms": 42}


def test_write_observations_jsonl_round_trips(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "vision.jsonl"
    observation = build_vision_observation(text="hello", image_hash="sha256:image")

    written = write_observations_jsonl(output, [observation])

    assert written == output
    reloaded = observations_from_jsonl(output)
    assert len(reloaded) == 1
    assert reloaded[0]["text_hash"] == observation["text_hash"]
