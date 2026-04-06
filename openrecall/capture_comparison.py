from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from openrecall.mcp_client import (
    build_envelope as itir_build_envelope,
    compare_observations as itir_compare_observations,
    score_coherence as itir_score_coherence,
)


@dataclass(frozen=True)
class CaptureComparison:
    text_similarity: float
    image_similarity: float
    time_gap_seconds: int | None
    changed: bool
    change_confidence: str
    shared_terms: tuple[str, ...]
    new_terms: tuple[str, ...]


def extract_text_from_image(image: np.ndarray) -> str:
    from openrecall.ocr import extract_text_from_image as _extract_text_from_image

    return _extract_text_from_image(image)


def get_embedding(text: str) -> np.ndarray:
    from openrecall.nlp import get_embedding as _get_embedding

    return _get_embedding(text)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    from openrecall.nlp import cosine_similarity as _cosine_similarity

    return _cosine_similarity(left, right)


def mean_structured_similarity_index(left: np.ndarray, right: np.ndarray) -> float:
    from openrecall.screenshot import (
        mean_structured_similarity_index as _mean_structured_similarity_index,
    )

    return float(_mean_structured_similarity_index(left, right))


def compare_captures(
    left_image: np.ndarray,
    right_image: np.ndarray,
    *,
    left_timestamp: int | None = None,
    right_timestamp: int | None = None,
) -> dict[str, Any]:
    left_text = extract_text_from_image(left_image)
    right_text = extract_text_from_image(right_image)
    return compare_capture_texts(
        left_text,
        right_text,
        left_image=left_image,
        right_image=right_image,
        left_timestamp=left_timestamp,
        right_timestamp=right_timestamp,
    )


def compare_capture_texts(
    left_text: str,
    right_text: str,
    *,
    left_image: np.ndarray | None = None,
    right_image: np.ndarray | None = None,
    left_timestamp: int | None = None,
    right_timestamp: int | None = None,
) -> dict[str, Any]:
    left_observation = _build_observation_payload(
        left_text,
        timestamp=left_timestamp,
    )
    right_observation = _build_observation_payload(
        right_text,
        timestamp=right_timestamp,
    )
    bridge_comparison = itir_compare_observations(left_observation, right_observation)
    bridge_signals = bridge_comparison.get("signals") if isinstance(bridge_comparison.get("signals"), Mapping) else {}
    if "text_overlap" in bridge_signals:
        text_similarity = float(bridge_signals.get("text_overlap") or 0.0)
    else:
        text_similarity = float(bridge_comparison.get("similarity") or 0.0)

    image_similarity = 0.0
    if left_image is not None and right_image is not None:
        image_similarity = _round_metric(
            float(mean_structured_similarity_index(left_image, right_image))
        )

    left_terms = _token_terms(left_text)
    right_terms = _token_terms(right_text)
    shared_terms = tuple(sorted(left_terms & right_terms)[:8])
    new_terms = tuple(sorted(right_terms - left_terms)[:8])

    time_gap_seconds = None
    if left_timestamp is not None and right_timestamp is not None:
        time_gap_seconds = abs(int(right_timestamp) - int(left_timestamp))

    change_score = _change_score(text_similarity, image_similarity, time_gap_seconds)
    changed = change_score >= 0.25
    result = CaptureComparison(
        text_similarity=text_similarity,
        image_similarity=image_similarity,
        time_gap_seconds=time_gap_seconds,
        changed=changed,
        change_confidence=_confidence_band(change_score),
        shared_terms=shared_terms,
        new_terms=new_terms,
    )
    return {
        "text_similarity": result.text_similarity,
        "image_similarity": result.image_similarity,
        "time_gap_seconds": result.time_gap_seconds,
        "changed": result.changed,
        "change_confidence": result.change_confidence,
        "shared_terms": list(result.shared_terms),
        "new_terms": list(result.new_terms),
        "itir_similarity": bridge_comparison.get("similarity"),
        "itir_confidence": bridge_comparison.get("confidence"),
    }


def score_timeline_change_window(
    captures: list[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(captures) < 2:
        return {
            "comparison_count": 0,
            "mean_change_score": 0.0,
            "change_confidence": "low",
            "largest_change": None,
        }

    observations = [
        _build_observation_payload(
            str(capture.get("text") or ""),
            timestamp=_coerce_int(capture.get("timestamp")),
        )
        for capture in captures
    ]
    coherence_payload = itir_score_coherence(observations)
    comparisons: list[dict[str, Any]] = []
    for index in range(len(captures) - 1):
        left = captures[index]
        right = captures[index + 1]
        comparisons.append(
            compare_capture_texts(
                str(left.get("text") or ""),
                str(right.get("text") or ""),
                left_timestamp=_coerce_int(left.get("timestamp")),
                right_timestamp=_coerce_int(right.get("timestamp")),
            )
        )

    change_scores = [_window_change_score(item) for item in comparisons]
    mean_change_score = _round_metric(sum(change_scores) / len(change_scores))
    largest_index = max(range(len(change_scores)), key=lambda idx: change_scores[idx])
    return {
        "comparison_count": len(comparisons),
        "mean_change_score": mean_change_score,
        "change_confidence": str(coherence_payload.get("confidence") or _confidence_band(mean_change_score)),
        "itir_coherence": coherence_payload.get("coherence"),
        "largest_change": {
            "index": largest_index,
            "comparison": comparisons[largest_index],
        },
    }


def build_capture_comparison_envelope(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    comparison: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    comparison_payload = comparison or compare_capture_texts(
        str(left.get("text") or ""),
        str(right.get("text") or ""),
        left_timestamp=_coerce_int(left.get("timestamp")),
        right_timestamp=_coerce_int(right.get("timestamp")),
    )
    left_payload = _capture_payload(left)
    right_payload = _capture_payload(right)
    bridge_envelope = itir_build_envelope(
        _build_observation_payload(
            str(left.get("text") or ""),
            timestamp=_coerce_int(left.get("timestamp")),
        ),
        _build_observation_payload(
            str(right.get("text") or ""),
            timestamp=_coerce_int(right.get("timestamp")),
        ),
    )
    return {
        "envelope_type": "openrecall.capture_comparison.v1",
        "left_capture": left_payload,
        "right_capture": right_payload,
        "comparison": dict(comparison_payload),
        "itir_envelope": bridge_envelope,
        "meta": {
            "generated_by": "openrecall.capture_comparison",
            "version": "v1",
        },
    }


def _capture_payload(capture: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "timestamp": _coerce_int(capture.get("timestamp")),
        "active_app_name": str(capture.get("active_app_name") or ""),
        "active_window_title": str(capture.get("active_window_title") or ""),
        "filename": str(capture.get("filename") or ""),
    }


def _build_observation_payload(text: str, *, timestamp: int | None) -> dict[str, Any]:
    observed_time = None
    if timestamp is not None:
        observed_time = datetime_from_timestamp(timestamp)
    return {
        "source_family": "openrecall",
        "source_system": "openrecall",
        "source_scope": "internal",
        "capture_mode": "screen_capture",
        "observed_time": observed_time,
        "text": text,
    }


def _token_terms(text: str) -> set[str]:
    return {token.lower() for token in str(text or "").split() if token.strip()}


def _change_score(
    text_similarity: float,
    image_similarity: float,
    time_gap_seconds: int | None,
) -> float:
    time_signal = 0.0
    if time_gap_seconds is not None:
        time_signal = min(float(time_gap_seconds) / 3600.0, 1.0)
    raw = ((1.0 - text_similarity) * 0.5) + ((1.0 - image_similarity) * 0.4) + (time_signal * 0.1)
    return _round_metric(max(0.0, min(raw, 1.0)))


def _window_change_score(comparison: Mapping[str, Any]) -> float:
    return _change_score(
        float(comparison.get("text_similarity") or 0.0),
        float(comparison.get("image_similarity") or 0.0),
        _coerce_int(comparison.get("time_gap_seconds")),
    )


def _confidence_band(value: float) -> str:
    if value >= 0.75:
        return "high"
    if value >= 0.4:
        return "medium"
    return "low"


def _round_metric(value: float) -> float:
    return round(float(value), 3)


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def datetime_from_timestamp(timestamp: int) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
