import json
import sys
from pathlib import Path

ROOT = Path("/home/c/Documents/code/ITIR-suite")
OPENRECALL_ROOT = ROOT / "openrecall"
sys.path.insert(0, str(OPENRECALL_ROOT / "openrecall"))

from browser_assist import (  # noqa: E402
    build_browser_assist_record,
    build_record_from_mcp_snapshot,
    main,
    parse_mcp_snapshot_text,
    write_records_jsonl,
)


MCP_SNAPSHOT = """### Page
- Page URL: https://example.test/private?q=secret
- Page Title: Private Page
### Snapshot
```yaml
- generic [active] [ref=e1]:
  - heading "Account settings" [level=1] [ref=e2]
  - link "Billing" [ref=e3]
  - button "Save" [ref=e4]
```
### Events
- New console entries: .playwright-mcp/console.log#L1
"""


def test_build_browser_assist_record_defaults_to_metadata_only() -> None:
    row = build_browser_assist_record(
        page_url="https://example.test/private?q=secret",
        page_title="Private Page",
        visible_text="alpha beta gamma",
        structure={"heading_count": 1, "heading_text": ["Private Heading"], "link_count": 2},
        task_label="check something",
        ts="2026-06-05T04:00:00Z",
    )

    dumped = json.dumps(row)
    assert row["version"] == "browser_assist_session_v1"
    assert row["storage_mode"] == "metadata_only"
    assert row["text_preview"] is None
    assert row["text_hash"].startswith("sha256:")
    assert row["page_url_hash"].startswith("sha256:")
    assert row["page_title_hash"].startswith("sha256:")
    assert "example.test" not in dumped
    assert "Private Page" not in dumped
    assert "alpha beta" not in dumped
    assert row["structure_summary"]["heading_count"] == 1
    assert row["structure_summary"]["heading_text_hash"].startswith("sha256:")
    assert row["kanban_projection_policy"] == "observer_only"
    assert row["non_authoritative"] is True
    assert row["pnf_candidates"] == []
    assert row["task_identity_residual"] is None
    assert row["lifecycle_residual"] is None
    assert "task_id" not in row
    assert "kanban_card_id" not in row
    assert "workflow_state" not in row


def test_build_browser_assist_record_can_include_bounded_preview() -> None:
    row = build_browser_assist_record(
        visible_text="alpha beta gamma delta",
        include_preview=True,
        preview_chars=12,
        ts="2026-06-05T04:00:00Z",
    )

    assert row["storage_mode"] == "preview_plus_hashes"
    assert row["text_preview"] == "alpha beta…"


def test_parse_mcp_snapshot_text_extracts_page_and_structure() -> None:
    parsed = parse_mcp_snapshot_text(MCP_SNAPSHOT)

    assert parsed["page_url"] == "https://example.test/private?q=secret"
    assert parsed["page_title"] == "Private Page"
    assert "Account settings" in parsed["visible_text"]
    assert parsed["structure"]["heading_count"] == 1
    assert parsed["structure"]["link_count"] == 1
    assert parsed["structure"]["form_control_count"] == 1


def test_parse_saved_mcp_snapshot_body_without_page_wrapper() -> None:
    parsed = parse_mcp_snapshot_text(
        """- generic [active] [ref=e1]:
  - navigation [ref=e2]
  - heading "Search Results" [level=1] [ref=e3]
  - main [ref=e4]
  - contentinfo [ref=e5]
"""
    )

    assert parsed["page_url"] is None
    assert "Search Results" in parsed["visible_text"]
    assert parsed["structure"]["heading_count"] == 1
    assert parsed["structure"]["landmark_count"] == 3
    assert parsed["structure"]["landmark_roles"] == ["contentinfo", "main", "navigation"]


def test_build_record_from_mcp_snapshot_hashes_and_refs() -> None:
    row = build_record_from_mcp_snapshot(
        MCP_SNAPSHOT,
        snapshot_ref=".playwright-mcp/snapshot.md",
        include_preview=False,
        ts="2026-06-05T04:00:00Z",
    )

    dumped = json.dumps(row)
    assert row["provenance"]["collector"] == "playwright_mcp_snapshot"
    assert row["playwright_snapshot_refs"] == [".playwright-mcp/snapshot.md"]
    assert row["storage_mode"] == "metadata_only"
    assert row["text_preview"] is None
    assert "example.test" not in dumped
    assert "Private Page" not in dumped


def test_write_records_jsonl_round_trips(tmp_path: Path) -> None:
    output = tmp_path / "browser_assist.jsonl"
    row = build_browser_assist_record(visible_text="hello", ts="2026-06-05T04:00:00Z")

    write_records_jsonl(output, [row])

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["text_hash"] == row["text_hash"]


def test_cli_normalizes_mcp_snapshot_file(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "snapshot.md"
    output = tmp_path / "raw.jsonl"
    source.write_text(MCP_SNAPSHOT, encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        ["browser_assist.py", "--mcp-snapshot-input", str(source), "--output", str(output)],
    )

    main()

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["provenance"]["collector"] == "playwright_mcp_snapshot"
    assert rows[0]["storage_mode"] == "metadata_only"
