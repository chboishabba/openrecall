from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


NORMALIZATION_VERSION = "openrecall.ocr_normalize.v1"

_WHITESPACE_RE = re.compile(r"[ \t]+")
_NEWLINE_ARTIFACT_RE = re.compile(r"(?<=\w)\s+(?:ln|In)\s+(?=\w)")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:/)])")
_SPACE_AFTER_OPEN_RE = re.compile(r"([(/`])\s+")
_DOUBLE_HYPHEN_SPACE_RE = re.compile(r"(\w)\s*-\s+(\w)")
_UNDERSCORE_SPACE_RE = re.compile(r"(\w)_\s+(\w)")
_DOT_SPACE_RE = re.compile(r"(\w)\.\s+(\w)")
_FLAG_SPACE_RE = re.compile(r"(--[A-Za-z0-9]+)\s+([A-Za-z0-9_-]+)")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_BACKTICK_ARTIFACT_RE = re.compile(r"[“”]")


@dataclass
class NormalizationResult:
    raw_text: str
    normalized_text: str
    version: str = NORMALIZATION_VERSION
    issue_taxonomy: dict[str, int] = field(default_factory=dict)

    def metadata(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "issue_taxonomy": dict(self.issue_taxonomy),
        }


def _count_subn(pattern: re.Pattern[str], repl: str, text: str) -> tuple[str, int]:
    updated, count = pattern.subn(repl, text)
    return updated, count


def normalize_ocr_text(raw_text: str) -> NormalizationResult:
    text = str(raw_text or "")
    issue_taxonomy = {
        "whitespace_newline_corruption": 0,
        "punctuation_corruption": 0,
        "token_splits_joins": 0,
        "code_token_corruption": 0,
        "extra_text": 0,
    }

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")

    normalized, count = _count_subn(_BACKTICK_ARTIFACT_RE, '"', normalized)
    issue_taxonomy["punctuation_corruption"] += count

    normalized, count = _count_subn(_NEWLINE_ARTIFACT_RE, "\n", normalized)
    issue_taxonomy["whitespace_newline_corruption"] += count

    normalized, count = _count_subn(_MULTI_NEWLINE_RE, "\n\n", normalized)
    issue_taxonomy["whitespace_newline_corruption"] += count

    lines = []
    for line in normalized.split("\n"):
        collapsed, collapsed_count = _count_subn(_WHITESPACE_RE, " ", line.strip())
        issue_taxonomy["whitespace_newline_corruption"] += collapsed_count
        lines.append(collapsed)
    normalized = "\n".join(lines).strip()

    normalized, count = _count_subn(_SPACE_BEFORE_PUNCT_RE, r"\1", normalized)
    issue_taxonomy["punctuation_corruption"] += count

    normalized, count = _count_subn(_SPACE_AFTER_OPEN_RE, r"\1", normalized)
    issue_taxonomy["punctuation_corruption"] += count

    normalized, count = _count_subn(_DOUBLE_HYPHEN_SPACE_RE, r"\1-\2", normalized)
    issue_taxonomy["token_splits_joins"] += count
    issue_taxonomy["code_token_corruption"] += count

    normalized, count = _count_subn(_UNDERSCORE_SPACE_RE, r"\1_\2", normalized)
    issue_taxonomy["token_splits_joins"] += count
    issue_taxonomy["code_token_corruption"] += count

    normalized, count = _count_subn(_DOT_SPACE_RE, r"\1.\2", normalized)
    issue_taxonomy["token_splits_joins"] += count
    issue_taxonomy["code_token_corruption"] += count

    normalized, count = _count_subn(_FLAG_SPACE_RE, r"\1-\2", normalized)
    issue_taxonomy["token_splits_joins"] += count
    issue_taxonomy["code_token_corruption"] += count

    issue_taxonomy = {
        key: value for key, value in issue_taxonomy.items() if value > 0 or key == "extra_text"
    }
    return NormalizationResult(
        raw_text=text,
        normalized_text=normalized,
        issue_taxonomy=issue_taxonomy,
    )
