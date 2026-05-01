import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/home/c/Documents/code/ITIR-suite")
OPENRECALL_ROOT = ROOT / "openrecall"
sys.path.insert(0, str(OPENRECALL_ROOT / "openrecall"))

import ocr
from ocr_normalize import normalize_ocr_text
from ocr_jobs import benchmark_manifest, process_image_paths


def test_extract_text_from_image_uses_http_backend(monkeypatch):
    calls = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "text": "remote text",
                    "segments": [{"text": "remote text"}],
                    "metadata": {"provider": "test"},
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        calls["timeout"] = timeout
        calls["url"] = request.full_url
        calls["headers"] = dict(request.header_items())
        body = json.loads(request.data.decode("utf-8"))
        calls["has_image"] = bool(body["image_base64"])
        return FakeResponse()

    monkeypatch.setenv("OPENRECALL_OCR_BACKEND", "http")
    monkeypatch.setenv("OPENRECALL_OCR_API_URL", "http://ocr.example/api")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result = ocr.extract_ocr_result_from_image(np.zeros((2, 2, 3), dtype=np.uint8))

    assert result.text == "remote text"
    assert result.backend == "http"
    assert result.segments == [{"text": "remote text"}]
    assert result.metadata == {"provider": "test"}
    assert calls["url"] == "http://ocr.example/api"
    assert calls["timeout"] == 60
    assert calls["has_image"] is True


def test_process_image_paths_uses_selected_backend(tmp_path, monkeypatch):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (2, 2), color="white").save(image_path)

    monkeypatch.setattr(
        "ocr_jobs.extract_ocr_result_from_image",
        lambda image, backend=None: ocr.OCRResult(
            text="offline text",
            backend=backend or "local",
            elapsed_seconds=0.25,
            segments=[{"text": "offline text"}],
        ),
    )

    records = process_image_paths([str(tmp_path)], backend="http")

    assert len(records) == 1
    assert records[0]["image_path"] == str(image_path)
    assert records[0]["backend"] == "http"
    assert records[0]["char_count"] == len("offline text")
    assert records[0]["raw_text"] == "offline text"
    assert records[0]["normalized_text"] == "offline text"


def test_normalize_ocr_text_repairs_code_shaped_artifacts():
    result = normalize_ocr_text(
        "captured_ date\nwrite- back\npython -m openrecall\nfoo. bar\n"
    )

    assert result.normalized_text == (
        "captured_date\nwrite-back\npython -m openrecall\nfoo.bar"
    )
    assert result.issue_taxonomy["token_splits_joins"] >= 1
    assert result.issue_taxonomy["code_token_corruption"] >= 1


def test_benchmark_manifest_reports_speed_and_accuracy(tmp_path, monkeypatch):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (2, 2), color="white").save(image_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps([{"image_path": str(image_path), "expected_text": "hello world"}])
    )

    monkeypatch.setattr(
        "ocr_jobs.extract_ocr_result_from_image",
        lambda image, backend=None: ocr.OCRResult(
            text="hello world",
            backend=backend or "local",
            elapsed_seconds=0.5,
        ),
    )

    result = benchmark_manifest(str(manifest_path), backend="local")

    assert result["image_count"] == 1
    assert result["backend"] == "local"
    assert result["chars_per_second"] == len("hello world") / 0.5
    assert result["raw_metrics"]["exact_match_rate"] == 1.0
    assert result["raw_metrics"]["char_error_rate"] == 0.0
    assert result["normalized_metrics"]["exact_match_rate"] == 1.0
    assert result["records"][0]["raw_diff"] == ""
    assert result["records"][0]["normalized_diff"] == ""
