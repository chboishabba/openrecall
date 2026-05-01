import base64
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol
from typing import Any


@dataclass
class OCRResult:
    text: str
    backend: str
    elapsed_seconds: float
    segments: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class OCRProvider(Protocol):
    name: str

    def extract(self, image) -> OCRResult:
        ...


def _resolve_ocr_backend(backend: str | None = None) -> str:
    resolved = (
        backend
        or os.environ.get("OPENRECALL_OCR_BACKEND")
        or ("http" if os.environ.get("OPENRECALL_OCR_API_URL") else "local")
    )
    resolved = resolved.lower()
    if resolved not in {"local", "http"}:
        raise ValueError(
            "Unsupported OCR backend. Expected 'local' or 'http', got "
            f"{resolved!r}."
        )
    return resolved


def _load_local_predictor():
    from doctr.models import ocr_predictor

    return ocr_predictor(
        pretrained=True,
        det_arch="db_mobilenet_v3_large",
        reco_arch="crnn_mobilenet_v3_large",
    )


_LOCAL_PREDICTOR = None


def _get_local_predictor():
    global _LOCAL_PREDICTOR
    if _LOCAL_PREDICTOR is None:
        _LOCAL_PREDICTOR = _load_local_predictor()
    return _LOCAL_PREDICTOR


def _http_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    token = os.environ.get("OPENRECALL_OCR_API_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    raw_headers = os.environ.get("OPENRECALL_OCR_API_HEADERS")
    if raw_headers:
        headers.update(json.loads(raw_headers))
    return headers


class LocalOCRProvider:
    name = "local"

    def extract(self, image) -> OCRResult:
        started = time.perf_counter()
        result = _get_local_predictor()([image])
        blocks: list[str] = []
        segments: list[dict[str, Any]] = []
        for page_index, page in enumerate(result.pages):
            for block_index, block in enumerate(page.blocks):
                for line_index, line in enumerate(block.lines):
                    line_words = [word.value for word in line.words]
                    if not line_words:
                        continue
                    text = " ".join(line_words)
                    blocks.append(text)
                    segments.append(
                        {
                            "page_index": page_index,
                            "block_index": block_index,
                            "line_index": line_index,
                            "text": text,
                            "word_count": len(line_words),
                        }
                    )
        return OCRResult(
            text="\n\n".join(blocks).strip(),
            backend=self.name,
            elapsed_seconds=time.perf_counter() - started,
            segments=segments,
        )


class HTTPOCRProvider:
    name = "http"

    def extract(self, image) -> OCRResult:
        url = os.environ.get("OPENRECALL_OCR_API_URL")
        if not url:
            raise RuntimeError(
                "OPENRECALL_OCR_API_URL is required when OPENRECALL_OCR_BACKEND=http."
            )

        import numpy as np
        from io import BytesIO
        from PIL import Image

        started = time.perf_counter()
        rgb_array = np.asarray(image, dtype=np.uint8)
        buffer = BytesIO()
        Image.fromarray(rgb_array).convert("RGB").save(buffer, format="PNG")
        payload = json.dumps(
            {
                "image_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
                "mime_type": "image/png",
            }
        ).encode("utf-8")

        request = urllib.request.Request(url, data=payload, headers=_http_headers())
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise RuntimeError(f"External OCR request failed: {exc}") from exc

        parsed = json.loads(body)
        return OCRResult(
            text=str(parsed.get("text", "")).strip(),
            backend=self.name,
            elapsed_seconds=time.perf_counter() - started,
            segments=list(parsed.get("segments") or []),
            metadata=dict(parsed.get("metadata") or {}),
        )


def _provider_map() -> dict[str, OCRProvider]:
    return {
        "local": LocalOCRProvider(),
        "http": HTTPOCRProvider(),
    }


def extract_ocr_result_from_image(image, backend: str | None = None) -> OCRResult:
    resolved = _resolve_ocr_backend(backend)
    return _provider_map()[resolved].extract(image)


def extract_text_from_image(image, backend: str | None = None) -> str:
    return extract_ocr_result_from_image(image, backend=backend).text
