import json
import sqlite3
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path("/home/c/Documents/code/ITIR-suite")
OPENRECALL_ROOT = ROOT / "openrecall"
sys.path.insert(0, str(OPENRECALL_ROOT))
sys.path.insert(0, str(OPENRECALL_ROOT / "openrecall"))

from bundle_jobs import (
    export_session_bundle,
    ingest_session_bundle,
    query_history_timeline,
    verify_manifest_signature,
    writeback_ocr_results,
)


def _create_test_db(db_path: Path, *, timestamp: int = 1714521600, text: str = "Daily report notes") -> None:
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                app TEXT,
                title TEXT,
                text TEXT,
                captured_date TEXT,
                timestamp INTEGER UNIQUE,
                embedding BLOB
            )
            """
        )
        cursor.execute(
            """
            INSERT INTO entries (app, title, text, captured_date, timestamp, embedding)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "Browser",
                "Inbox",
                text,
                "2024-05-01",
                timestamp,
                np.array([0.1, 0.2], dtype=np.float32).tobytes(),
            ),
        )
        conn.commit()


def test_export_session_bundle_creates_manifest_and_capture_archive(tmp_path: Path) -> None:
    db_path = tmp_path / "recall.db"
    screenshots_dir = tmp_path / "screenshots"
    screenshots_dir.mkdir()
    _create_test_db(db_path)
    Image.new("RGB", (2, 2), color="white").save(screenshots_dir / "1714521600_0.webp")

    output_path = tmp_path / "bundle.orbundle.zip"
    payload = export_session_bundle(
        date_str="2024-05-01",
        session_id="session-001",
        org_id="acme",
        user_id="alice",
        device_id="desktop-a",
        output_path=str(output_path),
        db_path=str(db_path),
        screenshots_path=str(screenshots_dir),
    )

    manifest = payload["manifest"]
    assert output_path.exists()
    assert manifest["entry_count"] == 1
    assert manifest["user_id"] == "alice"
    assert manifest["device_id"] == "desktop-a"
    assert manifest["entries"][0]["captured_date"] == "2024-05-01"
    assert manifest["entries"][0]["capture_files"] == ["1714521600_0.webp"]
    assert manifest["entries"][0]["captured_at"] == float(1714521600)

    with zipfile.ZipFile(output_path) as archive:
        names = set(archive.namelist())
        assert "manifest.json" in names
        assert "captures/1714521600_0.webp" in names


def test_export_session_bundle_can_sign_manifest_and_set_retention(tmp_path: Path) -> None:
    db_path = tmp_path / "recall.db"
    screenshots_dir = tmp_path / "screenshots"
    screenshots_dir.mkdir()
    _create_test_db(db_path)
    Image.new("RGB", (2, 2), color="white").save(screenshots_dir / "1714521600_0.webp")

    payload = export_session_bundle(
        date_str="2024-05-01",
        session_id="session-signed",
        org_id="acme",
        user_id="alice",
        device_id="desktop-a",
        output_path=str(tmp_path / "signed.orbundle.zip"),
        db_path=str(db_path),
        screenshots_path=str(screenshots_dir),
        retention_days=30,
        signing_secret="topsecret",
    )

    manifest = payload["manifest"]
    assert manifest["retention"]["retention_days"] == 30
    assert manifest["signature"]["algorithm"] == "hmac-sha256"
    assert verify_manifest_signature(manifest, "topsecret") is True


def test_ingest_session_bundle_tracks_multiple_sessions(tmp_path: Path) -> None:
    db_path = tmp_path / "recall.db"
    screenshots_dir = tmp_path / "screenshots"
    screenshots_dir.mkdir()
    _create_test_db(db_path)
    Image.new("RGB", (2, 2), color="white").save(screenshots_dir / "1714521600_0.webp")

    output_path = tmp_path / "bundle.orbundle.zip"
    export_session_bundle(
        date_str="2024-05-01",
        session_id="session-002",
        org_id="acme",
        user_id="bob",
        device_id="laptop-b",
        output_path=str(output_path),
        db_path=str(db_path),
        screenshots_path=str(screenshots_dir),
    )

    registry_root = tmp_path / "processing-server"
    payload = ingest_session_bundle(
        bundle_path=str(output_path),
        registry_root=str(registry_root),
    )

    registry_db = Path(payload["registry_db_path"])
    assert registry_db.exists()
    assert Path(payload["extract_path"]).exists()

    with sqlite3.connect(registry_db) as conn:
        bundle_row = conn.execute(
            "SELECT org_id, user_id, device_id, session_id, entry_count FROM bundles"
        ).fetchone()
        assert bundle_row == ("acme", "bob", "laptop-b", "session-002", 1)
        entry_row = conn.execute(
            "SELECT captured_date, app, title, text, capture_files_json FROM bundle_entries"
        ).fetchone()
        assert entry_row[0:4] == ("2024-05-01", "Browser", "Inbox", "Daily report notes")
        assert json.loads(entry_row[4]) == ["1714521600_0.webp"]


def test_ingest_session_bundle_verifies_signed_manifest(tmp_path: Path) -> None:
    db_path = tmp_path / "recall.db"
    screenshots_dir = tmp_path / "screenshots"
    screenshots_dir.mkdir()
    _create_test_db(db_path)
    Image.new("RGB", (2, 2), color="white").save(screenshots_dir / "1714521600_0.webp")

    output_path = tmp_path / "signed.orbundle.zip"
    export_session_bundle(
        date_str="2024-05-01",
        session_id="session-verified",
        org_id="acme",
        user_id="bob",
        device_id="laptop-b",
        output_path=str(output_path),
        db_path=str(db_path),
        screenshots_path=str(screenshots_dir),
        retention_days=7,
        signing_secret="topsecret",
    )

    payload = ingest_session_bundle(
        bundle_path=str(output_path),
        registry_root=str(tmp_path / "registry"),
        signing_secret="topsecret",
        source_mode="pull",
    )
    assert payload["signature_valid"] is True
    assert payload["source_mode"] == "pull"

    with sqlite3.connect(Path(payload["registry_db_path"])) as conn:
        row = conn.execute(
            "SELECT delete_after, signature_algorithm, signature_valid, source_mode FROM bundles"
        ).fetchone()
        assert row[0] is not None
        assert row[1] == "hmac-sha256"
        assert row[2] == 1
        assert row[3] == "pull"


def test_query_history_timeline_keeps_device_provenance_and_filters(tmp_path: Path) -> None:
    registry_root = tmp_path / "registry"

    db_path_a = tmp_path / "recall-a.db"
    screenshots_dir_a = tmp_path / "screens-a"
    screenshots_dir_a.mkdir()
    _create_test_db(db_path_a, timestamp=1714521600, text="Desktop event")
    Image.new("RGB", (2, 2), color="white").save(screenshots_dir_a / "1714521600_0.webp")

    db_path_b = tmp_path / "recall-b.db"
    screenshots_dir_b = tmp_path / "screens-b"
    screenshots_dir_b.mkdir()
    _create_test_db(db_path_b, timestamp=1714521602, text="Laptop event")
    Image.new("RGB", (2, 2), color="white").save(screenshots_dir_b / "1714521602_0.webp")

    bundle_a = tmp_path / "bundle-a.zip"
    bundle_b = tmp_path / "bundle-b.zip"
    export_session_bundle(
        date_str="2024-05-01",
        session_id="session-a",
        org_id="acme",
        user_id="alice",
        device_id="desktop-a",
        output_path=str(bundle_a),
        db_path=str(db_path_a),
        screenshots_path=str(screenshots_dir_a),
    )
    export_session_bundle(
        date_str="2024-05-01",
        session_id="session-b",
        org_id="acme",
        user_id="alice",
        device_id="laptop-b",
        output_path=str(bundle_b),
        db_path=str(db_path_b),
        screenshots_path=str(screenshots_dir_b),
    )
    ingest_session_bundle(bundle_path=str(bundle_a), registry_root=str(registry_root))
    ingest_session_bundle(bundle_path=str(bundle_b), registry_root=str(registry_root))

    combined = query_history_timeline(
        registry_root=str(registry_root),
        org_id="acme",
        user_id="alice",
    )
    assert [row["device_id"] for row in combined] == ["desktop-a", "laptop-b"]
    assert [row["text"] for row in combined] == ["Desktop event", "Laptop event"]

    filtered = query_history_timeline(
        registry_root=str(registry_root),
        org_id="acme",
        user_id="alice",
        device_ids=["laptop-b"],
    )
    assert len(filtered) == 1
    assert filtered[0]["device_id"] == "laptop-b"
    assert filtered[0]["text"] == "Laptop event"
    assert filtered[0]["captured_date"] == "2024-05-01"

    day_filtered = query_history_timeline(
        registry_root=str(registry_root),
        org_id="acme",
        user_id="alice",
        captured_date="2024-05-01",
    )
    assert len(day_filtered) == 2


def test_writeback_ocr_results_populates_local_recall_db(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "recall.db"
    screenshots_dir = tmp_path / "screenshots"
    screenshots_dir.mkdir()
    _create_test_db(db_path)
    Image.new("RGB", (2, 2), color="white").save(screenshots_dir / "1714521600_0.webp")

    bundle_path = tmp_path / "bundle.orbundle.zip"
    export_payload = export_session_bundle(
        date_str="2024-05-01",
        session_id="session-003",
        org_id="acme",
        user_id="charlie",
        device_id="desktop-c",
        output_path=str(bundle_path),
        db_path=str(db_path),
        screenshots_path=str(screenshots_dir),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(export_payload["manifest"]))
    ocr_results_path = tmp_path / "ocr_results.json"
    ocr_results_path.write_text(
        json.dumps(
            [
                {
                    "image_path": str(screenshots_dir / "1714521600_0.webp"),
                    "text": "Processed remotely",
                    "backend": "http",
                    "elapsed_seconds": 0.2,
                    "embedding": [0.5] * 384,
                    "segments": [{"label": "line", "text": "Processed remotely"}],
                    "metadata": {"processor": "secure-cloud"},
                }
            ]
        )
    )

    import bundle_jobs as jobs
    import database as local_database

    monkeypatch.setattr(jobs, "default_db_path", str(db_path))
    monkeypatch.setattr(local_database, "db_path", str(db_path))

    payload = writeback_ocr_results(
        manifest_path=str(manifest_path),
        ocr_results_path=str(ocr_results_path),
        db_path=str(db_path),
    )

    assert payload["written_entry_count"] == 1
    assert payload["missing_capture_count"] == 0
    assert payload["remote_embedding_count"] == 1
    assert payload["vision_payload_count"] == 1

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT text, app, title, captured_date, embedding FROM entries WHERE timestamp = ?",
            (1714521600,),
        ).fetchone()
        assert row[0] == "Processed remotely"
        assert row[1] == "Browser"
        assert row[2] == "Inbox"
        assert row[3] == "2024-05-01"
        restored = np.frombuffer(row[4], dtype=np.float32)
        assert restored.shape == (384,)
        assert np.allclose(restored, np.array([0.5] * 384, dtype=np.float32))

    vision_path = Path(payload["vision_payload_dir"]) / "1714521600.vision.json"
    vision_payload = json.loads(vision_path.read_text())
    assert vision_payload["user_id"] == "charlie"
    assert vision_payload["device_id"] == "desktop-c"
    assert vision_payload["segments"] == [{"label": "line", "text": "Processed remotely"}]
    assert vision_payload["metadata"] == [{"processor": "secure-cloud"}]
