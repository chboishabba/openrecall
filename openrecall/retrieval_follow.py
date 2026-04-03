"""Bounded retrieval/follow helpers for OpenRecall."""

from __future__ import annotations

from typing import Any, Iterable

try:
    from .normalized_artifact import (
        build_search_follow_artifact,
        build_search_follow_normalized_artifact,
    )
except ImportError:  # pragma: no cover - allows direct module import in focused tests
    from normalized_artifact import (  # type: ignore
        build_search_follow_artifact,
        build_search_follow_normalized_artifact,
    )


def build_retrieval_follow_sidecar(
    query: str,
    results: Iterable[Any],
    *,
    max_results: int = 5,
    stop_after: bool = True,
) -> dict[str, object]:
    """Build the raw bounded retrieval/follow artifact for an OpenRecall query."""

    return build_search_follow_artifact(
        query,
        results,
        max_results=max_results,
        stop_after=stop_after,
    )


def build_retrieval_follow_normalized_artifact(
    query: str,
    results: Iterable[Any],
    *,
    max_results: int = 5,
    stop_after: bool = True,
    profile_version: str = "openrecall.search_follow.v1",
) -> dict[str, object]:
    """Build the suite-normalized derived_product wrapper for a bounded retrieval run."""

    sidecar = build_retrieval_follow_sidecar(
        query,
        results,
        max_results=max_results,
        stop_after=stop_after,
    )
    return build_search_follow_normalized_artifact(
        sidecar,
        profile_version=profile_version,
    )
