from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest


ROOT = Path(__file__).resolve().parents[2]
OPENRECALL_ROOT = ROOT / "openrecall"
sys.path.insert(0, str(OPENRECALL_ROOT / "openrecall"))

import config as openrecall_config

pytest.importorskip("flask")


def test_entry_detail_route_uses_stable_entry_id(tmp_path: Path) -> None:
    storage_root = tmp_path / "storage"
    screenshots_dir = storage_root / "screenshots"
    screenshots_dir.mkdir(parents=True)
    db_path = storage_root / "recall.db"

    openrecall_config.db_path = str(db_path)
    openrecall_config.screenshots_path = str(screenshots_dir)

    import database as openrecall_database
    import app as openrecall_app

    openrecall_database.db_path = str(db_path)
    openrecall_app.screenshots_path = str(screenshots_dir)
    openrecall_database.create_db()

    timestamp = 1714521600
    Image.new("RGB", (2, 2), color="white").save(screenshots_dir / f"{timestamp}_0.webp")
    entry_id = openrecall_database.insert_entry(
        text="raw text",
        timestamp=timestamp,
        embedding=np.zeros(4, dtype=np.float32),
        app="terminal",
        title="bash - session",
        captured_date="2024-05-01",
        normalized_text="normalized text",
        normalization_version="openrecall.ocr_normalize.v1",
        normalization_issues_json="[]",
    )
    assert entry_id is not None

    client = openrecall_app.app.test_client()
    response = client.get(f"/entry/{entry_id}")
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert f"OpenRecall Entry {entry_id}" in body
    assert "bash - session" in body
    assert "normalized text" in body
    assert f"/static/{timestamp}_0.webp" in body


def test_entry_detail_route_404s_for_missing_entry() -> None:
    import app as openrecall_app

    client = openrecall_app.app.test_client()
    response = client.get("/entry/999999")
    assert response.status_code == 404
