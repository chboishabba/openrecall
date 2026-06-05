"""Read-only browser-assist observations for OpenRecall-adjacent workflows."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping
from uuid import uuid4

VERSION = "browser_assist_session_v1"
DEFAULT_PREVIEW_CHARS = 240
DEFAULT_CDP_URL = "http://127.0.0.1:9222"


_PAGE_SNAPSHOT_SCRIPT = r"""
() => {
  const visible = (el) => {
    const style = window.getComputedStyle(el);
    return style && style.visibility !== 'hidden' && style.display !== 'none';
  };
  const texts = [];
  const walker = document.createTreeWalker(document.body || document.documentElement, NodeFilter.SHOW_TEXT);
  while (walker.nextNode()) {
    const node = walker.currentNode;
    const text = (node.nodeValue || '').replace(/\s+/g, ' ').trim();
    if (text && node.parentElement && visible(node.parentElement)) {
      texts.push(text);
    }
  }
  const headingText = Array.from(document.querySelectorAll('h1,h2,h3,h4,h5,h6'))
    .filter(visible)
    .map((el) => (el.innerText || el.textContent || '').replace(/\s+/g, ' ').trim())
    .filter(Boolean);
  const landmarks = Array.from(document.querySelectorAll('main,nav,header,footer,aside,section,[role]'))
    .filter(visible)
    .map((el) => (el.getAttribute('role') || el.tagName || '').toLowerCase())
    .filter(Boolean);
  return {
    url: window.location.href,
    title: document.title || '',
    visible_text: texts.join(' '),
    structure: {
      heading_count: headingText.length,
      heading_text: headingText.slice(0, 24),
      link_count: document.querySelectorAll('a[href]').length,
      form_control_count: document.querySelectorAll('input,textarea,select,button').length,
      image_count: document.images ? document.images.length : 0,
      landmark_count: landmarks.length,
      landmark_roles: Array.from(new Set(landmarks)).slice(0, 24)
    }
  };
}
"""


def _now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _collapse_ws(value: Any) -> str:
    return " ".join(str(value or "").split())


def _truncate(value: Any, max_chars: int) -> str | None:
    text = _collapse_ws(value)
    if not text:
        return None
    limit = max(1, int(max_chars))
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _hash_sequence(values: Iterable[Any]) -> str | None:
    clean = [_collapse_ws(value) for value in values]
    clean = [value for value in clean if value]
    if not clean:
        return None
    return sha256_text(json.dumps(clean, ensure_ascii=False, sort_keys=True))


def _bounded_structure_summary(structure: Mapping[str, Any] | None) -> dict[str, Any]:
    raw = dict(structure or {})
    heading_text = [_collapse_ws(item) for item in raw.get("heading_text") or []]
    landmark_roles = [_collapse_ws(item).lower() for item in raw.get("landmark_roles") or []]
    return {
        "heading_count": int(raw.get("heading_count") or 0),
        "heading_text_hash": _hash_sequence(heading_text),
        "heading_preview": [_truncate(item, 80) for item in heading_text[:5] if _collapse_ws(item)],
        "link_count": int(raw.get("link_count") or 0),
        "form_control_count": int(raw.get("form_control_count") or 0),
        "image_count": int(raw.get("image_count") or 0),
        "landmark_count": int(raw.get("landmark_count") or 0),
        "landmark_roles": sorted({role for role in landmark_roles if role})[:12],
    }


def parse_mcp_snapshot_text(text: str) -> dict[str, Any]:
    """Extract bounded page fields from a Playwright MCP snapshot response."""

    page_url = None
    page_title = None
    lines = str(text or "").splitlines()
    snapshot_lines: list[str] = []
    in_snapshot = False
    in_code_fence = False

    for raw_line in lines:
        line = raw_line.rstrip()
        if line.startswith("- Page URL:"):
            page_url = line.partition(":")[2].strip()
            continue
        if line.startswith("- Page Title:"):
            page_title = line.partition(":")[2].strip()
            continue
        if line.strip() == "### Snapshot":
            in_snapshot = True
            continue
        if in_snapshot and line.strip().startswith("```"):
            in_code_fence = not in_code_fence
            continue
        if in_snapshot and in_code_fence:
            snapshot_lines.append(line)
        elif in_snapshot and line.strip().startswith("### "):
            break

    if not snapshot_lines:
        snapshot_lines = [
            line.rstrip()
            for line in lines
            if line.strip() and not line.lstrip().startswith("### ") and not line.lstrip().startswith("- Page ")
        ]

    snapshot_text = "\n".join(snapshot_lines).strip()
    landmark_role_names = {
        "banner",
        "complementary",
        "contentinfo",
        "form",
        "main",
        "navigation",
        "region",
        "search",
    }
    landmark_roles = []
    for line in snapshot_lines:
        clean = line.lstrip()
        if not clean.startswith("- "):
            continue
        role = clean[2:].split(" ", 1)[0].split("[", 1)[0].strip().lower().rstrip(":")
        if role in landmark_role_names:
            landmark_roles.append(role)
    heading_count = sum(1 for line in snapshot_lines if line.lstrip().startswith("- heading"))
    link_count = sum(1 for line in snapshot_lines if line.lstrip().startswith("- link"))
    form_control_count = sum(
        1
        for line in snapshot_lines
        if line.lstrip().startswith(("- button", "- textbox", "- checkbox", "- radio", "- combobox"))
    )
    return {
        "page_url": page_url,
        "page_title": page_title,
        "visible_text": snapshot_text,
        "structure": {
            "heading_count": heading_count,
            "heading_text": [],
            "link_count": link_count,
            "form_control_count": form_control_count,
            "image_count": sum(1 for line in snapshot_lines if line.lstrip().startswith("- image")),
            "landmark_count": len(landmark_roles),
            "landmark_roles": sorted(set(landmark_roles)),
        },
    }


def build_browser_assist_record(
    *,
    page_url: str | None = None,
    page_title: str | None = None,
    visible_text: str | None = None,
    structure: Mapping[str, Any] | None = None,
    session_id: str | None = None,
    task_label: str | None = None,
    ts: str | None = None,
    browser: str = "chrome",
    mode: str = "observe",
    include_preview: bool = False,
    preview_chars: int = DEFAULT_PREVIEW_CHARS,
    screenshot_refs: Iterable[str] | None = None,
    playwright_snapshot_refs: Iterable[str] | None = None,
    openrecall_entry_refs: Iterable[str] | None = None,
    transcript_refs: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Build a bounded, non-authoritative browser observation row.

    Full URL, title, visible text, and HTML remain producer-owned and are not
    persisted in this record. The row is shaped for StatiBaker's browser-assist
    adapter while keeping OpenRecall responsible for capture quality.
    """

    row_ts = ts or _now_utc()
    text = _collapse_ws(visible_text)
    storage_mode = "preview_plus_hashes" if include_preview else "metadata_only"
    record = {
        "version": VERSION,
        "ts": row_ts,
        "session_id": session_id or f"browser-assist:{uuid4()}",
        "task_label": _truncate(task_label, 120),
        "mode": mode,
        "browser": browser,
        "started_at": row_ts,
        "ended_at": None,
        "page_url_hash": sha256_text(page_url) if page_url else None,
        "page_title_hash": sha256_text(page_title) if page_title else None,
        "text_preview": _truncate(text, preview_chars) if include_preview and text else None,
        "text_hash": sha256_text(text) if text else None,
        "structure_summary": _bounded_structure_summary(structure),
        "openrecall_entry_refs": list(openrecall_entry_refs or []),
        "playwright_snapshot_refs": list(playwright_snapshot_refs or []),
        "screenshot_refs": list(screenshot_refs or []),
        "transcript_refs": list(transcript_refs or []),
        "pnf_candidates": [],
        "task_identity_residual": None,
        "lifecycle_residual": None,
        "kanban_projection_policy": "observer_only",
        "storage_mode": storage_mode,
        "non_authoritative": True,
        "bounded": True,
        "provenance": {
            "source": "openrecall.browser_assist",
            "collector": "playwright_cdp",
            "collected_at": row_ts,
        },
    }
    record["artifact_hash"] = sha256_text(json.dumps(record, sort_keys=True, ensure_ascii=False))
    return record


def build_record_from_mcp_snapshot(
    snapshot_text: str,
    *,
    snapshot_ref: str | None = None,
    session_id: str | None = None,
    task_label: str | None = None,
    ts: str | None = None,
    include_preview: bool = False,
    preview_chars: int = DEFAULT_PREVIEW_CHARS,
) -> dict[str, Any]:
    parsed = parse_mcp_snapshot_text(snapshot_text)
    record = build_browser_assist_record(
        page_url=parsed.get("page_url"),
        page_title=parsed.get("page_title"),
        visible_text=parsed.get("visible_text"),
        structure=parsed.get("structure"),
        session_id=session_id,
        task_label=task_label,
        ts=ts,
        browser="chrome",
        include_preview=include_preview,
        preview_chars=preview_chars,
        playwright_snapshot_refs=[snapshot_ref] if snapshot_ref else [],
    )
    record["provenance"]["collector"] = "playwright_mcp_snapshot"
    if snapshot_ref:
        record["provenance"]["snapshot_ref"] = snapshot_ref
    record["artifact_hash"] = sha256_text(json.dumps(record, sort_keys=True, ensure_ascii=False))
    return record


async def collect_page_record(
    page: Any,
    *,
    session_id: str,
    task_label: str | None = None,
    include_preview: bool = False,
    preview_chars: int = DEFAULT_PREVIEW_CHARS,
    artifact_dir: str | Path | None = None,
    screenshot: bool = False,
) -> dict[str, Any]:
    snapshot = await page.evaluate(_PAGE_SNAPSHOT_SCRIPT)
    ts = _now_utc()
    screenshot_refs: list[str] = []
    snapshot_refs: list[str] = []

    artifact_root = Path(artifact_dir) if artifact_dir else None
    if artifact_root:
        artifact_root.mkdir(parents=True, exist_ok=True)
        snapshot_path = artifact_root / f"{session_id.replace(':', '-')}-{int(time.time() * 1000)}.structure.json"
        bounded_snapshot = {
            "ts": ts,
            "page_url_hash": sha256_text(str(snapshot.get("url") or "")) if snapshot.get("url") else None,
            "page_title_hash": sha256_text(str(snapshot.get("title") or "")) if snapshot.get("title") else None,
            "text_hash": sha256_text(_collapse_ws(snapshot.get("visible_text"))) if snapshot.get("visible_text") else None,
            "structure_summary": _bounded_structure_summary(snapshot.get("structure")),
            "non_authoritative": True,
            "bounded": True,
        }
        snapshot_path.write_text(json.dumps(bounded_snapshot, sort_keys=True) + "\n", encoding="utf-8")
        snapshot_refs.append(str(snapshot_path))

    if screenshot and artifact_root:
        screenshot_path = artifact_root / f"{session_id.replace(':', '-')}-{int(time.time() * 1000)}.png"
        await page.screenshot(path=str(screenshot_path), full_page=False)
        screenshot_refs.append(f"{screenshot_path}#{sha256_file(screenshot_path)}")

    return build_browser_assist_record(
        page_url=snapshot.get("url"),
        page_title=snapshot.get("title"),
        visible_text=snapshot.get("visible_text"),
        structure=snapshot.get("structure"),
        session_id=session_id,
        task_label=task_label,
        ts=ts,
        include_preview=include_preview,
        preview_chars=preview_chars,
        screenshot_refs=screenshot_refs,
        playwright_snapshot_refs=snapshot_refs,
    )


def write_records_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]], *, append: bool = False) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with target.open(mode, encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), sort_keys=True) + "\n")
    return target


def normalize_mcp_snapshot_file(
    *,
    input_path: str | Path,
    output: str | Path,
    task_label: str | None = None,
    include_preview: bool = False,
    preview_chars: int = DEFAULT_PREVIEW_CHARS,
    append: bool = False,
) -> Path:
    source = Path(input_path)
    record = build_record_from_mcp_snapshot(
        source.read_text(encoding="utf-8"),
        snapshot_ref=str(source),
        task_label=task_label,
        include_preview=include_preview,
        preview_chars=preview_chars,
    )
    return write_records_jsonl(output, [record], append=append)


async def collect_from_chrome_cdp(
    *,
    cdp_url: str,
    output: str | Path,
    task_label: str | None = None,
    once: bool = False,
    duration_seconds: float | None = None,
    interval_seconds: float = 5.0,
    include_preview: bool = False,
    preview_chars: int = DEFAULT_PREVIEW_CHARS,
    artifact_dir: str | Path | None = None,
    screenshot: bool = False,
    page_index: int = -1,
    append: bool = False,
) -> Path:
    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:  # pragma: no cover - exercised without importing playwright.
        raise RuntimeError(
            "Playwright is required for browser assist capture. "
            "Install with `pip install -e .[browser-assist]` and run `playwright install chromium`."
        ) from exc

    session_id = f"browser-assist:{uuid4()}"
    records: list[dict[str, Any]] = []
    started = time.monotonic()

    async with async_playwright() as playwright:
        browser = await playwright.chromium.connect_over_cdp(cdp_url)
        try:
            while True:
                contexts = list(getattr(browser, "contexts", []) or [])
                pages = [page for context in contexts for page in list(getattr(context, "pages", []) or [])]
                if not pages:
                    raise RuntimeError("No Chrome pages are available through the CDP connection.")
                page = pages[page_index]
                records.append(
                    await collect_page_record(
                        page,
                        session_id=session_id,
                        task_label=task_label,
                        include_preview=include_preview,
                        preview_chars=preview_chars,
                        artifact_dir=artifact_dir,
                        screenshot=screenshot,
                    )
                )
                if once:
                    break
                if duration_seconds is not None and time.monotonic() - started >= duration_seconds:
                    break
                await asyncio.sleep(max(0.1, interval_seconds))
        finally:
            # This collector attaches to a user-owned Chrome. Do not close the
            # browser process; exiting the Playwright context drops our client.
            pass

    return write_records_jsonl(output, records, append=append)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Capture read-only Chrome browser-assist observations.")
    parser.add_argument("--cdp-url", default=DEFAULT_CDP_URL, help="Chrome DevTools Protocol URL")
    parser.add_argument("--mcp-snapshot-input", default=None, help="Normalize a saved Playwright MCP snapshot file")
    parser.add_argument("--output", required=True, help="Raw browser-assist JSONL output path")
    parser.add_argument("--task-label", default=None, help="Optional bounded task label")
    parser.add_argument("--once", action="store_true", help="Capture one observation and exit")
    parser.add_argument("--duration-seconds", type=float, default=None, help="Stop after this many seconds")
    parser.add_argument("--interval-seconds", type=float, default=5.0, help="Seconds between observations")
    parser.add_argument("--include-preview", action="store_true", help="Persist bounded text previews")
    parser.add_argument("--preview-chars", type=int, default=DEFAULT_PREVIEW_CHARS, help="Maximum preview chars")
    parser.add_argument("--artifact-dir", default=None, help="Optional producer-owned local artifact directory")
    parser.add_argument("--screenshot", action="store_true", help="Capture screenshot refs under --artifact-dir")
    parser.add_argument("--page-index", type=int, default=-1, help="Page index from CDP contexts; default last page")
    parser.add_argument("--append", action="store_true", help="Append to output instead of replacing it")
    args = parser.parse_args(argv)

    if args.mcp_snapshot_input:
        normalize_mcp_snapshot_file(
            input_path=args.mcp_snapshot_input,
            output=args.output,
            task_label=args.task_label,
            include_preview=bool(args.include_preview),
            preview_chars=max(1, args.preview_chars),
            append=bool(args.append),
        )
        return

    if args.screenshot and not args.artifact_dir:
        parser.error("--screenshot requires --artifact-dir")
    if not args.once and args.duration_seconds is None:
        parser.error("use --once or --duration-seconds")

    asyncio.run(
        collect_from_chrome_cdp(
            cdp_url=args.cdp_url,
            output=args.output,
            task_label=args.task_label,
            once=bool(args.once),
            duration_seconds=args.duration_seconds,
            interval_seconds=max(0.1, args.interval_seconds),
            include_preview=bool(args.include_preview),
            preview_chars=max(1, args.preview_chars),
            artifact_dir=args.artifact_dir,
            screenshot=bool(args.screenshot),
            page_index=args.page_index,
            append=bool(args.append),
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
