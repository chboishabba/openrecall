import argparse
import hashlib
import hmac
import json
import socket
import sqlite3
import zipfile
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

import numpy as np

try:
    from openrecall.config import db_path as default_db_path
    from openrecall.config import screenshots_path as default_screenshots_path
except ModuleNotFoundError:  # pragma: no cover - vendored test path shim
    from config import db_path as default_db_path
    from config import screenshots_path as default_screenshots_path

try:
    from openrecall.database import create_db, upsert_entry
    from openrecall.nlp import EMBEDDING_DIM, get_embedding
    import openrecall.database as openrecall_database
except ModuleNotFoundError:  # pragma: no cover - vendored test path shim
    from database import create_db, upsert_entry
    from nlp import EMBEDDING_DIM, get_embedding
    import database as openrecall_database


@dataclass
class BundleWindow:
    start_timestamp: int
    end_timestamp: int
    captured_date: str


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sign_manifest_payload(payload: dict[str, Any], secret: str) -> str:
    return hmac.new(
        secret.encode("utf-8"),
        _canonical_json(payload),
        hashlib.sha256,
    ).hexdigest()


def _split_manifest_for_signing(
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    payload = dict(manifest)
    signature = payload.pop("signature", None)
    return payload, signature if isinstance(signature, dict) else None


def verify_manifest_signature(manifest: dict[str, Any], secret: str) -> bool:
    payload, signature = _split_manifest_for_signing(manifest)
    if signature is None:
        return False
    expected = _sign_manifest_payload(payload, secret)
    return hmac.compare_digest(str(signature.get("hmac_sha256") or ""), expected)


def build_day_window(date_str: str) -> BundleWindow:
    captured_date = date.fromisoformat(date_str)
    start_dt = datetime.combine(captured_date, time.min, tzinfo=UTC)
    end_dt = datetime.combine(captured_date, time.max, tzinfo=UTC)
    return BundleWindow(
        start_timestamp=int(start_dt.timestamp()),
        end_timestamp=int(end_dt.timestamp()),
        captured_date=date_str,
    )


def query_entries_for_window(
    db_path: str,
    *,
    captured_date: str,
    start_timestamp: int,
    end_timestamp: int,
) -> list[dict[str, Any]]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        columns = {
            row[1] for row in cursor.execute("PRAGMA table_info(entries)").fetchall()
        }
        if "captured_date" not in columns:
            cursor.execute("ALTER TABLE entries ADD COLUMN captured_date TEXT")
            cursor.execute(
                """
                UPDATE entries
                SET captured_date = date(timestamp, 'unixepoch')
                WHERE captured_date IS NULL AND timestamp IS NOT NULL
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_captured_date_timestamp
                ON entries (captured_date, timestamp)
                """
            )
            conn.commit()
        cursor.execute(
            """
            SELECT id, app, title, text, captured_date, timestamp
            FROM entries
            WHERE captured_date = ? AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp ASC
            """,
            (captured_date, start_timestamp, end_timestamp),
        )
        return [dict(row) for row in cursor.fetchall()]


def _find_capture_files(screenshots_dir: Path, timestamp: int) -> list[Path]:
    matches = sorted(screenshots_dir.glob(f"{timestamp}*.webp"))
    if matches:
        return matches
    for suffix in (".png", ".jpg", ".jpeg", ".bmp"):
        matches = sorted(screenshots_dir.glob(f"{timestamp}*{suffix}"))
        if matches:
            return matches
    return []


def _manifest_entries(
    entries: Iterable[dict[str, Any]],
    screenshots_dir: Path,
) -> list[dict[str, Any]]:
    manifest_entries: list[dict[str, Any]] = []
    for entry in entries:
        timestamp = int(entry["timestamp"])
        capture_files = _find_capture_files(screenshots_dir, timestamp)
        manifest_entries.append(
            {
                "entry_id": int(entry["id"]),
                "captured_date": str(entry.get("captured_date") or ""),
                "timestamp": timestamp,
                "captured_at": float(timestamp),
                "app": str(entry.get("app") or ""),
                "title": str(entry.get("title") or ""),
                "text": str(entry.get("text") or ""),
                "capture_files": [path.name for path in capture_files],
                "capture_count": len(capture_files),
            }
        )
    return manifest_entries


def _retention_payload(
    captured_date: str,
    retention_days: int | None,
) -> dict[str, Any] | None:
    if retention_days is None:
        return None
    delete_after = (
        datetime.combine(date.fromisoformat(captured_date), time.min, tzinfo=UTC)
        + timedelta(days=int(retention_days))
    )
    return {
        "retention_days": int(retention_days),
        "delete_after": delete_after.isoformat(),
    }


def build_session_bundle_manifest(
    *,
    date_str: str,
    session_id: str,
    org_id: str,
    user_id: str,
    device_id: str,
    host_id: str | None = None,
    db_path: str = default_db_path,
    screenshots_path: str = default_screenshots_path,
    retention_days: int | None = None,
) -> dict[str, Any]:
    window = build_day_window(date_str)
    screenshots_dir = Path(screenshots_path)
    entries = query_entries_for_window(
        db_path,
        captured_date=window.captured_date,
        start_timestamp=window.start_timestamp,
        end_timestamp=window.end_timestamp,
    )
    manifest_entries = _manifest_entries(entries, screenshots_dir)
    capture_files = sorted(
        {capture for entry in manifest_entries for capture in entry["capture_files"]}
    )
    manifest = {
        "schema_version": "openrecall.session_bundle.v1",
        "bundle_id": f"openrecall.bundle:{uuid4()}",
        "captured_date": window.captured_date,
        "session_id": session_id,
        "org_id": org_id,
        "user_id": user_id,
        "device_id": device_id,
        "host_id": host_id or socket.gethostname(),
        "source": {
            "db_path": str(db_path),
            "screenshots_path": str(screenshots_path),
        },
        "created_at": datetime.now(UTC).isoformat(),
        "entry_count": len(manifest_entries),
        "capture_file_count": len(capture_files),
        "entries": manifest_entries,
    }
    retention = _retention_payload(window.captured_date, retention_days)
    if retention is not None:
        manifest["retention"] = retention
    return manifest


def _normalize_manifest_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(manifest)
    if "user_id" not in normalized and "employee_id" in normalized:
        normalized["user_id"] = normalized["employee_id"]
    if "device_id" not in normalized:
        normalized["device_id"] = str(normalized.get("host_id") or "unknown-device")
    return normalized


def export_session_bundle(
    *,
    date_str: str,
    session_id: str,
    org_id: str,
    user_id: str,
    device_id: str,
    output_path: str,
    host_id: str | None = None,
    db_path: str = default_db_path,
    screenshots_path: str = default_screenshots_path,
    compression: str = "lzma",
    retention_days: int | None = None,
    signing_secret: str | None = None,
) -> dict[str, Any]:
    manifest = build_session_bundle_manifest(
        date_str=date_str,
        session_id=session_id,
        org_id=org_id,
        user_id=user_id,
        device_id=device_id,
        host_id=host_id,
        db_path=db_path,
        screenshots_path=screenshots_path,
        retention_days=retention_days,
    )
    if signing_secret:
        manifest["signature"] = {
            "algorithm": "hmac-sha256",
            "hmac_sha256": _sign_manifest_payload(manifest, signing_secret),
        }

    compression_mode = {
        "lzma": zipfile.ZIP_LZMA,
        "deflate": zipfile.ZIP_DEFLATED,
        "stored": zipfile.ZIP_STORED,
    }[compression]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    screenshots_dir = Path(screenshots_path)

    with zipfile.ZipFile(output, "w", compression=compression_mode) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
        for entry in manifest["entries"]:
            for capture_name in entry["capture_files"]:
                capture_path = screenshots_dir / capture_name
                if capture_path.exists():
                    archive.write(capture_path, arcname=f"captures/{capture_name}")
    return {"bundle_path": str(output), "manifest": manifest}


def _ensure_registry_schema(registry_db_path: Path) -> None:
    with sqlite3.connect(registry_db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS bundles (
                bundle_id TEXT PRIMARY KEY,
                org_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                device_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                captured_date TEXT NOT NULL,
                host_id TEXT,
                source_mode TEXT NOT NULL DEFAULT 'push',
                archive_path TEXT NOT NULL,
                extract_path TEXT NOT NULL,
                imported_at TEXT NOT NULL,
                delete_after TEXT,
                signature_algorithm TEXT,
                signature_valid INTEGER NOT NULL DEFAULT 0,
                entry_count INTEGER NOT NULL,
                capture_file_count INTEGER NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS bundle_entries (
                bundle_id TEXT NOT NULL,
                org_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                device_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                captured_date TEXT NOT NULL,
                entry_id INTEGER NOT NULL,
                timestamp INTEGER NOT NULL,
                captured_at REAL NOT NULL,
                app TEXT,
                title TEXT,
                text TEXT,
                capture_files_json TEXT NOT NULL,
                vision_payload_json TEXT,
                PRIMARY KEY (bundle_id, entry_id),
                FOREIGN KEY(bundle_id) REFERENCES bundles(bundle_id)
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS history_events (
                history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                bundle_id TEXT NOT NULL,
                org_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                device_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                captured_date TEXT NOT NULL,
                entry_id INTEGER NOT NULL,
                timestamp INTEGER NOT NULL,
                captured_at REAL NOT NULL,
                app TEXT,
                title TEXT,
                text TEXT,
                capture_files_json TEXT NOT NULL,
                vision_payload_json TEXT,
                imported_at TEXT NOT NULL,
                source_mode TEXT NOT NULL DEFAULT 'push',
                UNIQUE(bundle_id, entry_id),
                FOREIGN KEY(bundle_id) REFERENCES bundles(bundle_id)
            )
            """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_history_scope_time ON history_events (org_id, user_id, captured_at)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_history_device_time ON history_events (org_id, user_id, device_id, captured_at)"
        )
        bundle_entry_columns = {
            row[1] for row in cursor.execute("PRAGMA table_info(bundle_entries)").fetchall()
        }
        if "captured_date" not in bundle_entry_columns:
            cursor.execute(
                "ALTER TABLE bundle_entries ADD COLUMN captured_date TEXT NOT NULL DEFAULT ''"
            )
        history_columns = {
            row[1] for row in cursor.execute("PRAGMA table_info(history_events)").fetchall()
        }
        if "captured_date" not in history_columns:
            cursor.execute(
                "ALTER TABLE history_events ADD COLUMN captured_date TEXT NOT NULL DEFAULT ''"
            )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_history_scope_day_time
            ON history_events (org_id, user_id, captured_date, timestamp)
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_history_device_day_time
            ON history_events (org_id, user_id, device_id, captured_date, timestamp)
            """
        )
        conn.commit()


def _bundle_extract_dir(registry_dir: Path, manifest: dict[str, Any]) -> Path:
    return (
        registry_dir
        / "sessions"
        / manifest["org_id"]
        / manifest["user_id"]
        / manifest["device_id"]
        / manifest["session_id"]
        / manifest["bundle_id"].replace(":", "_")
    )


def ingest_session_bundle(
    *,
    bundle_path: str,
    registry_root: str,
    signing_secret: str | None = None,
    source_mode: str = "push",
) -> dict[str, Any]:
    bundle_file = Path(bundle_path)
    registry_dir = Path(registry_root)
    registry_dir.mkdir(parents=True, exist_ok=True)
    registry_db = registry_dir / "bundle_registry.sqlite"
    _ensure_registry_schema(registry_db)

    with zipfile.ZipFile(bundle_file) as archive:
        manifest = _normalize_manifest_identity(
            json.loads(archive.read("manifest.json").decode("utf-8"))
        )
        extract_dir = _bundle_extract_dir(registry_dir, manifest)
        extract_dir.mkdir(parents=True, exist_ok=True)
        archive.extractall(extract_dir)

    signature_valid = 0
    if manifest.get("signature") and signing_secret:
        signature_valid = 1 if verify_manifest_signature(manifest, signing_secret) else 0
        if not signature_valid:
            raise ValueError("Manifest signature verification failed.")

    imported_at = datetime.now(UTC).isoformat()
    with sqlite3.connect(registry_db) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO bundles (
                bundle_id, org_id, user_id, device_id, session_id, captured_date,
                host_id, source_mode, archive_path, extract_path, imported_at,
                delete_after, signature_algorithm, signature_valid, entry_count,
                capture_file_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                manifest["bundle_id"],
                manifest["org_id"],
                manifest["user_id"],
                manifest["device_id"],
                manifest["session_id"],
                manifest["captured_date"],
                manifest.get("host_id"),
                source_mode,
                str(bundle_file),
                str(extract_dir),
                imported_at,
                (manifest.get("retention") or {}).get("delete_after"),
                (manifest.get("signature") or {}).get("algorithm"),
                signature_valid,
                int(manifest["entry_count"]),
                int(manifest["capture_file_count"]),
            ),
        )
        cursor.execute("DELETE FROM bundle_entries WHERE bundle_id = ?", (manifest["bundle_id"],))
        cursor.execute("DELETE FROM history_events WHERE bundle_id = ?", (manifest["bundle_id"],))

        for entry in manifest["entries"]:
            vision_payload_path = extract_dir / "vision_payloads" / f"{int(entry['timestamp'])}.vision.json"
            vision_payload_json = (
                vision_payload_path.read_text() if vision_payload_path.exists() else None
            )
            captured_at = float(entry.get("captured_at") or entry["timestamp"])
            captured_date = str(
                entry.get("captured_date") or manifest.get("captured_date") or ""
            )
            row = (
                manifest["bundle_id"],
                manifest["org_id"],
                manifest["user_id"],
                manifest["device_id"],
                manifest["session_id"],
                captured_date,
                int(entry["entry_id"]),
                int(entry["timestamp"]),
                captured_at,
                entry.get("app"),
                entry.get("title"),
                entry.get("text"),
                json.dumps(entry.get("capture_files") or []),
                vision_payload_json,
            )
            cursor.execute(
                """
                INSERT INTO bundle_entries (
                    bundle_id, org_id, user_id, device_id, session_id, captured_date, entry_id,
                    timestamp, captured_at, app, title, text, capture_files_json,
                    vision_payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                row,
            )
            cursor.execute(
                """
                INSERT INTO history_events (
                    bundle_id, org_id, user_id, device_id, session_id, captured_date, entry_id,
                    timestamp, captured_at, app, title, text, capture_files_json,
                    vision_payload_json, imported_at, source_mode
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                row + (imported_at, source_mode),
            )
        conn.commit()

    return {
        "bundle_id": manifest["bundle_id"],
        "registry_db_path": str(registry_db),
        "extract_path": str(extract_dir),
        "signature_valid": bool(signature_valid) if manifest.get("signature") else None,
        "entry_count": int(manifest["entry_count"]),
        "capture_file_count": int(manifest["capture_file_count"]),
        "source_mode": source_mode,
    }


def _basename_map(ocr_results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for result in ocr_results:
        image_path = result.get("image_path")
        if image_path:
            mapping[Path(str(image_path)).name] = result
    return mapping


def _coerce_remote_embedding(remote_embedding: Any) -> np.ndarray | None:
    if remote_embedding is None:
        return None
    vector = np.asarray(remote_embedding, dtype=np.float32)
    if vector.shape != (EMBEDDING_DIM,):
        return None
    return vector


def writeback_ocr_results(
    *,
    manifest_path: str,
    ocr_results_path: str,
    db_path: str = default_db_path,
) -> dict[str, Any]:
    manifest = _normalize_manifest_identity(json.loads(Path(manifest_path).read_text()))
    ocr_results = json.loads(Path(ocr_results_path).read_text())
    if not isinstance(ocr_results, list):
        raise ValueError("OCR results must be a JSON array.")

    openrecall_database.db_path = str(db_path)
    create_db()
    results_by_name = _basename_map(ocr_results)
    manifest_dir = Path(manifest_path).parent
    vision_dir = manifest_dir / "vision_payloads"
    vision_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    missing: list[dict[str, Any]] = []
    remote_embedding_count = 0
    vision_payload_count = 0

    for entry in manifest.get("entries", []):
        texts: list[str] = []
        collected_segments: list[dict[str, Any]] = []
        collected_metadata: list[dict[str, Any]] = []
        embedding_vector: np.ndarray | None = None

        for capture_name in entry.get("capture_files", []):
            result = results_by_name.get(str(capture_name))
            if result is None:
                missing.append(
                    {
                        "entry_id": entry.get("entry_id"),
                        "timestamp": entry.get("timestamp"),
                        "capture_file": capture_name,
                    }
                )
                continue

            text = str(result.get("text") or "").strip()
            if text:
                texts.append(text)

            segments = list(result.get("segments") or [])
            metadata = dict(result.get("metadata") or {})
            if segments:
                collected_segments.extend(segments)
            if metadata:
                collected_metadata.append(metadata)

            maybe_embedding = _coerce_remote_embedding(result.get("embedding"))
            if maybe_embedding is not None and embedding_vector is None:
                embedding_vector = maybe_embedding
                remote_embedding_count += 1

        combined_text = "\n\n".join(texts).strip()
        if not combined_text:
            continue

        embedding = embedding_vector if embedding_vector is not None else get_embedding(combined_text)
        upsert_entry(
            combined_text,
            int(entry["timestamp"]),
            embedding,
            str(entry.get("app") or ""),
            str(entry.get("title") or ""),
            captured_date=str(
                entry.get("captured_date") or manifest.get("captured_date") or ""
            ),
        )

        if collected_segments or collected_metadata:
            vision_payload = {
                "bundle_id": manifest.get("bundle_id"),
                "org_id": manifest.get("org_id"),
                "user_id": manifest.get("user_id"),
                "device_id": manifest.get("device_id"),
                "session_id": manifest.get("session_id"),
                "entry_id": entry.get("entry_id"),
                "timestamp": entry.get("timestamp"),
                "capture_files": list(entry.get("capture_files") or []),
                "segments": collected_segments,
                "metadata": collected_metadata,
            }
            vision_path = vision_dir / f"{int(entry['timestamp'])}.vision.json"
            vision_path.write_text(json.dumps(vision_payload, indent=2, sort_keys=True) + "\n")
            vision_payload_count += 1
        written += 1

    return {
        "bundle_id": manifest.get("bundle_id"),
        "db_path": str(db_path),
        "written_entry_count": written,
        "remote_embedding_count": remote_embedding_count,
        "vision_payload_count": vision_payload_count,
        "vision_payload_dir": str(vision_dir),
        "missing_capture_count": len(missing),
        "missing_captures": missing,
    }


def query_history_timeline(
    *,
    registry_root: str,
    org_id: str,
    user_id: str,
    device_ids: list[str] | None = None,
    captured_date: str | None = None,
    captured_after: float | None = None,
    captured_before: float | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    registry_db = Path(registry_root) / "bundle_registry.sqlite"
    _ensure_registry_schema(registry_db)
    clauses = ["org_id = ?", "user_id = ?"]
    params: list[Any] = [org_id, user_id]

    if device_ids:
        placeholders = ",".join("?" for _ in device_ids)
        clauses.append(f"device_id IN ({placeholders})")
        params.extend(device_ids)
    if captured_date is not None:
        clauses.append("captured_date = ?")
        params.append(str(captured_date))
    if captured_after is not None:
        clauses.append("captured_at >= ?")
        params.append(float(captured_after))
    if captured_before is not None:
        clauses.append("captured_at <= ?")
        params.append(float(captured_before))

    sql = """
        SELECT bundle_id, org_id, user_id, device_id, session_id, captured_date, entry_id,
               timestamp, captured_at, app, title, text, capture_files_json,
               vision_payload_json, imported_at, source_mode
        FROM history_events
        WHERE {where}
        ORDER BY captured_at ASC, device_id ASC, entry_id ASC
    """.format(where=" AND ".join(clauses))
    if limit is not None:
        sql += " LIMIT ?"
        params.append(int(limit))

    with sqlite3.connect(registry_db) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenRecall session bundle tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser(
        "export-day",
        help="Export one day/session into a compressed bundle for remote processing.",
    )
    export_parser.add_argument("--date", required=True, help="Capture date as YYYY-MM-DD.")
    export_parser.add_argument("--session-id", required=True)
    export_parser.add_argument("--org-id", required=True)
    export_parser.add_argument("--user-id", required=True)
    export_parser.add_argument("--device-id", required=True)
    export_parser.add_argument("--output", required=True)
    export_parser.add_argument("--host-id", default=None)
    export_parser.add_argument("--db-path", default=default_db_path)
    export_parser.add_argument("--screenshots-path", default=default_screenshots_path)
    export_parser.add_argument("--retention-days", type=int, default=None)
    export_parser.add_argument(
        "--signing-secret",
        default=None,
        help="HMAC secret used to sign the bundle manifest.",
    )
    export_parser.add_argument(
        "--compression",
        choices=["lzma", "deflate", "stored"],
        default="lzma",
    )

    ingest_parser = subparsers.add_parser(
        "ingest-bundle",
        help="Ingest a session bundle into a central processing registry.",
    )
    ingest_parser.add_argument("--bundle", required=True)
    ingest_parser.add_argument("--registry-root", required=True)
    ingest_parser.add_argument(
        "--signing-secret",
        default=None,
        help="Verify the bundle manifest with this HMAC secret.",
    )
    ingest_parser.add_argument(
        "--source-mode",
        choices=["push", "pull"],
        default="push",
        help="Record whether this bundle arrived via client push or server pull.",
    )

    timeline_parser = subparsers.add_parser(
        "query-timeline",
        help="Query the shared central timeline for one org/user scope.",
    )
    timeline_parser.add_argument("--registry-root", required=True)
    timeline_parser.add_argument("--org-id", required=True)
    timeline_parser.add_argument("--user-id", required=True)
    timeline_parser.add_argument("--device-id", action="append", dest="device_ids")
    timeline_parser.add_argument("--captured-date", default=None)
    timeline_parser.add_argument("--captured-after", type=float, default=None)
    timeline_parser.add_argument("--captured-before", type=float, default=None)
    timeline_parser.add_argument("--limit", type=int, default=None)

    writeback_parser = subparsers.add_parser(
        "writeback-ocr",
        help="Write offline OCR results back into a local recall.db using a session manifest.",
    )
    writeback_parser.add_argument("--manifest", required=True)
    writeback_parser.add_argument("--ocr-results", required=True)
    writeback_parser.add_argument("--db-path", default=default_db_path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "export-day":
        payload = export_session_bundle(
            date_str=args.date,
            session_id=args.session_id,
            org_id=args.org_id,
            user_id=args.user_id,
            device_id=args.device_id,
            output_path=args.output,
            host_id=args.host_id,
            db_path=args.db_path,
            screenshots_path=args.screenshots_path,
            compression=args.compression,
            retention_days=args.retention_days,
            signing_secret=args.signing_secret,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.command == "ingest-bundle":
        payload = ingest_session_bundle(
            bundle_path=args.bundle,
            registry_root=args.registry_root,
            signing_secret=args.signing_secret,
            source_mode=args.source_mode,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.command == "query-timeline":
        payload = query_history_timeline(
            registry_root=args.registry_root,
            org_id=args.org_id,
            user_id=args.user_id,
            device_ids=args.device_ids,
            captured_date=args.captured_date,
            captured_after=args.captured_after,
            captured_before=args.captured_before,
            limit=args.limit,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.command == "writeback-ocr":
        payload = writeback_ocr_results(
            manifest_path=args.manifest,
            ocr_results_path=args.ocr_results,
            db_path=args.db_path,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
