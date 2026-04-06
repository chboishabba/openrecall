import numpy as np
import pytest

from openrecall.capture_comparison import (
    build_capture_comparison_envelope,
    compare_capture_texts,
    compare_captures,
    score_timeline_change_window,
)


def test_compare_capture_texts_uses_existing_embedding_path(monkeypatch):
    calls = {"bridge": []}

    def fake_itir_compare_observations(left, right):
        calls["bridge"].append((left, right))
        return {
            "similarity": 0.6,
            "confidence": "medium",
            "signals": {"text_overlap": 0.0, "time_proximity": 0.5, "geo_proximity": None},
        }

    monkeypatch.setattr(
        "openrecall.capture_comparison.itir_compare_observations",
        fake_itir_compare_observations,
    )

    result = compare_capture_texts(
        "Daily report draft",
        "Completely different screen",
        left_timestamp=100,
        right_timestamp=460,
    )

    assert len(calls["bridge"]) == 1
    assert result["text_similarity"] == 0.0
    assert result["time_gap_seconds"] == 360
    assert result["changed"] is True
    assert result["itir_similarity"] == 0.6
    assert "daily" not in result["shared_terms"]


def test_compare_captures_uses_existing_ocr_and_ssim(monkeypatch):
    left_image = np.zeros((2, 2, 3), dtype=np.uint8)
    right_image = np.ones((2, 2, 3), dtype=np.uint8)

    ocr_calls = []
    ssim_calls = []

    def fake_extract_text_from_image(image):
        ocr_calls.append(image.shape)
        return "shared note" if len(ocr_calls) == 1 else "shared note updated"

    def fake_mean_structured_similarity_index(left, right):
        ssim_calls.append((left.shape, right.shape))
        return 0.8

    monkeypatch.setattr(
        "openrecall.capture_comparison.extract_text_from_image",
        fake_extract_text_from_image,
    )
    monkeypatch.setattr(
        "openrecall.capture_comparison.itir_compare_observations",
        lambda left, right: {
            "similarity": 0.5,
            "confidence": "medium",
            "signals": {"text_overlap": 0.5, "time_proximity": 1.0, "geo_proximity": None},
        },
    )
    monkeypatch.setattr(
        "openrecall.capture_comparison.mean_structured_similarity_index",
        fake_mean_structured_similarity_index,
    )

    result = compare_captures(
        left_image,
        right_image,
        left_timestamp=100,
        right_timestamp=130,
    )

    assert ocr_calls == [(2, 2, 3), (2, 2, 3)]
    assert ssim_calls == [((2, 2, 3), (2, 2, 3))]
    assert result["text_similarity"] == 0.5
    assert result["image_similarity"] == 0.8
    assert result["shared_terms"] == ["note", "shared"]
    assert result["new_terms"] == ["updated"]


def test_score_timeline_change_window_tracks_largest_change(monkeypatch):
    def fake_compare_capture_texts(left_text, right_text, **kwargs):
        changed = left_text != right_text
        return {
            "text_similarity": 0.2 if changed else 1.0,
            "image_similarity": 0.1 if changed else 1.0,
            "time_gap_seconds": kwargs["right_timestamp"] - kwargs["left_timestamp"],
            "changed": changed,
            "change_confidence": "high" if changed else "low",
            "shared_terms": [],
            "new_terms": [right_text] if changed else [],
        }

    monkeypatch.setattr(
        "openrecall.capture_comparison.compare_capture_texts",
        fake_compare_capture_texts,
    )
    monkeypatch.setattr(
        "openrecall.capture_comparison.itir_score_coherence",
        lambda observations: {"coherence": 0.55, "confidence": "medium"},
    )

    result = score_timeline_change_window(
        [
            {"text": "alpha", "timestamp": 100},
            {"text": "alpha", "timestamp": 160},
            {"text": "beta", "timestamp": 220},
        ]
    )

    assert result["comparison_count"] == 2
    assert result["itir_coherence"] == 0.55
    assert result["largest_change"]["index"] == 1
    assert result["largest_change"]["comparison"]["new_terms"] == ["beta"]


def test_build_capture_comparison_envelope_keeps_boundary_small(monkeypatch):
    bridge_payload = {
        "version": "itir.build_envelope.v1",
        "envelope_type": "itir.observation_comparison.v1",
        "meta": {"generated_by": "itir.compare_observations"},
    }
    monkeypatch.setattr(
        "openrecall.capture_comparison.itir_build_envelope",
        lambda left, right: bridge_payload,
    )
    envelope = build_capture_comparison_envelope(
        {
            "timestamp": 100,
            "active_app_name": "Editor",
            "active_window_title": "draft.md",
            "filename": "100.webp",
            "text": "ignored source text",
        },
        {
            "timestamp": 160,
            "active_app_name": "Browser",
            "active_window_title": "docs",
            "filename": "160.webp",
            "text": "ignored target text",
        },
        comparison={
            "text_similarity": 0.6,
            "image_similarity": 0.7,
            "time_gap_seconds": 60,
            "changed": True,
            "change_confidence": "medium",
            "shared_terms": ["docs"],
            "new_terms": ["browser"],
        },
    )

    assert envelope["envelope_type"] == "openrecall.capture_comparison.v1"
    assert envelope["meta"]["generated_by"] == "openrecall.capture_comparison"
    assert "text" not in envelope["left_capture"]
    assert envelope["right_capture"]["filename"] == "160.webp"
    assert envelope["itir_envelope"]["envelope_type"] == "itir.observation_comparison.v1"


def test_itir_client_compare_requires_verified_guarded_result(monkeypatch):
    from openrecall import itir_client

    monkeypatch.setattr(
        "openrecall.itir_client.safe_call_tool",
        lambda name, payload: {
            "decision": "rejected",
            "status_explanation": {
                "primary_reason_code": "sensitive_data_request",
                "reason_codes": ["sensitive_data_request", "social_engineering"],
            },
            "result": None,
        },
    )

    with pytest.raises(itir_client.ItirBridgeError, match="sensitive_data_request"):
        itir_client.compare_observations(
            {"source_system": "openrecall", "text": "left"},
            {"source_system": "openrecall", "text": "right"},
        )
