import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

try:
    from openrecall.ocr import OCRResult, extract_ocr_result_from_image
except ModuleNotFoundError:  # pragma: no cover - vendored test path shim
    from ocr import OCRResult, extract_ocr_result_from_image


def _levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            insertion = current[right_index - 1] + 1
            deletion = previous[right_index] + 1
            substitution = previous[right_index - 1] + (left_char != right_char)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def _normalized_similarity(left: str, right: str) -> float:
    span = max(len(left), len(right))
    if span == 0:
        return 1.0
    return 1.0 - (_levenshtein_distance(left, right) / span)


def _iter_image_paths(paths: list[str]) -> Iterable[Path]:
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
                    yield child
        elif path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            yield path


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def process_image_paths(
    paths: list[str],
    backend: str | None = None,
) -> list[dict]:
    records = []
    for image_path in _iter_image_paths(paths):
        result = extract_ocr_result_from_image(_load_rgb(image_path), backend=backend)
        records.append(
            {
                "image_path": str(image_path),
                "backend": result.backend,
                "elapsed_seconds": result.elapsed_seconds,
                "text": result.text,
                "char_count": len(result.text),
                "segment_count": len(result.segments),
                "segments": result.segments,
                "metadata": result.metadata,
            }
        )
    return records


def benchmark_manifest(
    manifest_path: str,
    backend: str | None = None,
) -> dict:
    manifest = json.loads(Path(manifest_path).read_text())
    records = []
    total_duration = 0.0
    total_chars = 0
    total_similarity = 0.0
    total_expected_chars = 0
    total_char_distance = 0
    exact_matches = 0

    for item in manifest:
        image_path = Path(item["image_path"])
        expected_text = str(item.get("expected_text", ""))
        result: OCRResult = extract_ocr_result_from_image(
            _load_rgb(image_path),
            backend=backend,
        )
        actual_text = result.text
        distance = _levenshtein_distance(actual_text, expected_text)
        similarity = _normalized_similarity(actual_text, expected_text)
        total_duration += result.elapsed_seconds
        total_chars += len(actual_text)
        total_similarity += similarity
        total_expected_chars += len(expected_text)
        total_char_distance += distance
        if actual_text == expected_text:
            exact_matches += 1
        records.append(
            {
                "image_path": str(image_path),
                "backend": result.backend,
                "elapsed_seconds": result.elapsed_seconds,
                "actual_text": actual_text,
                "expected_text": expected_text,
                "char_count": len(actual_text),
                "expected_char_count": len(expected_text),
                "char_distance": distance,
                "normalized_similarity": similarity,
            }
        )

    image_count = len(records)
    return {
        "image_count": image_count,
        "backend": records[0]["backend"] if records else backend or "local",
        "total_duration_seconds": total_duration,
        "chars_per_second": (total_chars / total_duration) if total_duration else 0.0,
        "mean_normalized_similarity": (
            total_similarity / image_count if image_count else 0.0
        ),
        "exact_match_rate": (exact_matches / image_count) if image_count else 0.0,
        "char_error_rate": (
            total_char_distance / total_expected_chars if total_expected_chars else 0.0
        ),
        "records": records,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenRecall OCR offline tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    process_parser = subparsers.add_parser(
        "process",
        help="Process saved screenshots/images offline and emit JSON.",
    )
    process_parser.add_argument("paths", nargs="+", help="Image files or directories.")
    process_parser.add_argument(
        "--backend",
        choices=["local", "http"],
        default=None,
        help="Override OCR backend selection.",
    )
    process_parser.add_argument(
        "--output",
        default="-",
        help="Write JSON output to this path, or '-' for stdout.",
    )

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Benchmark OCR speed and accuracy against a manifest.",
    )
    benchmark_parser.add_argument(
        "--manifest",
        required=True,
        help="JSON array of {image_path, expected_text}.",
    )
    benchmark_parser.add_argument(
        "--backend",
        choices=["local", "http"],
        default=None,
        help="Override OCR backend selection.",
    )
    benchmark_parser.add_argument(
        "--output",
        default="-",
        help="Write JSON output to this path, or '-' for stdout.",
    )

    return parser


def _write_output(payload: dict | list[dict], output_path: str) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if output_path == "-":
        print(encoded)
        return
    Path(output_path).write_text(encoded + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "process":
        _write_output(process_image_paths(args.paths, backend=args.backend), args.output)
        return 0

    if args.command == "benchmark":
        _write_output(benchmark_manifest(args.manifest, backend=args.backend), args.output)
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
