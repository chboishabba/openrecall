import sqlite3
from collections import namedtuple
from datetime import UTC, datetime
import numpy as np
from typing import Any, List, Optional, Tuple

try:
    from openrecall.config import db_path
except ModuleNotFoundError:  # pragma: no cover - vendored test path shim
    from config import db_path

# Define the structure of a database entry using namedtuple
Entry = namedtuple(
    "Entry",
    ["id", "app", "title", "text", "captured_date", "timestamp", "embedding"],
)


def _captured_date_from_timestamp(timestamp: int) -> str:
    return datetime.fromtimestamp(int(timestamp), tz=UTC).date().isoformat()


def create_db() -> None:
    """
    Creates the SQLite database and the 'entries' table if they don't exist.

    The table schema includes columns for an auto-incrementing ID, application name,
    window title, extracted text, captured_date, timestamp, and text embedding.
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """CREATE TABLE IF NOT EXISTS entries (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       app TEXT,
                       title TEXT,
                       text TEXT,
                       captured_date TEXT,
                       timestamp INTEGER UNIQUE,
                       embedding BLOB
                   )"""
            )
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
            # Add index on timestamp for faster lookups
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_timestamp ON entries (timestamp)"
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_captured_date_timestamp
                ON entries (captured_date, timestamp)
                """
            )
            conn.commit()
    except sqlite3.Error as e:
        print(f"Database error during table creation: {e}")


def get_all_entries() -> List[Entry]:
    """
    Retrieves all entries from the database.

    Returns:
        List[Entry]: A list of all entries as Entry namedtuples.
                     Returns an empty list if the table is empty or an error occurs.
    """
    entries: List[Entry] = []
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row  # Return rows as dictionary-like objects
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, app, title, text, captured_date, timestamp, embedding
                FROM entries
                ORDER BY timestamp DESC
                """
            )
            results = cursor.fetchall()
            for row in results:
                # Deserialize the embedding blob back into a NumPy array
                embedding = np.frombuffer(row["embedding"], dtype=np.float32) # Assuming float32, adjust if needed
                entries.append(
                    Entry(
                        id=row["id"],
                        app=row["app"],
                        title=row["title"],
                        text=row["text"],
                        captured_date=row["captured_date"],
                        timestamp=row["timestamp"],
                        embedding=embedding,
                    )
                )
    except sqlite3.Error as e:
        print(f"Database error while fetching all entries: {e}")
    return entries


def get_timestamps() -> List[int]:
    """
    Retrieves all timestamps from the database, ordered descending.

    Returns:
        List[int]: A list of all timestamps.
                   Returns an empty list if the table is empty or an error occurs.
    """
    timestamps: List[int] = []
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            # Use the index for potentially faster retrieval
            cursor.execute("SELECT timestamp FROM entries ORDER BY timestamp DESC")
            results = cursor.fetchall()
            timestamps = [result[0] for result in results]
    except sqlite3.Error as e:
        print(f"Database error while fetching timestamps: {e}")
    return timestamps


def insert_entry(
    text: str,
    timestamp: int,
    embedding: np.ndarray,
    app: str,
    title: str,
    captured_date: str | None = None,
) -> Optional[int]:
    """
    Inserts a new entry into the database.

    Args:
        text (str): The extracted text content.
        timestamp (int): The Unix timestamp of the screenshot.
        embedding (np.ndarray): The embedding vector for the text.
        app (str): The name of the active application.
        title (str): The title of the active window.

    Returns:
        Optional[int]: The ID of the newly inserted row, or None if insertion fails.
                       Prints an error message to stderr on failure.
    """
    embedding_bytes: bytes = embedding.astype(np.float32).tobytes() # Ensure consistent dtype
    last_row_id: Optional[int] = None
    normalized_captured_date = captured_date or _captured_date_from_timestamp(timestamp)
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO entries (text, captured_date, timestamp, embedding, app, title)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(timestamp) DO NOTHING""", # Avoid duplicates based on timestamp
                (
                    text,
                    normalized_captured_date,
                    timestamp,
                    embedding_bytes,
                    app,
                    title,
                ),
            )
            conn.commit()
            if cursor.rowcount > 0: # Check if insert actually happened
                last_row_id = cursor.lastrowid
            # else:
                # Optionally log that a duplicate timestamp was encountered
                # print(f"Skipped inserting entry with duplicate timestamp: {timestamp}")

    except sqlite3.Error as e:
        # More specific error handling can be added (e.g., IntegrityError for UNIQUE constraint)
        print(f"Database error during insertion: {e}")
    return last_row_id


def upsert_entry(
    text: str,
    timestamp: int,
    embedding: np.ndarray,
    app: str,
    title: str,
    captured_date: str | None = None,
) -> Optional[int]:
    """
    Inserts or replaces an entry keyed by timestamp.

    This is used by offline write-back flows where capture metadata may already
    exist locally and the OCR/embedding payload arrives later from a deferred or
    remote processing job.
    """
    embedding_bytes: bytes = embedding.astype(np.float32).tobytes()
    last_row_id: Optional[int] = None
    normalized_captured_date = captured_date or _captured_date_from_timestamp(timestamp)
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO entries (text, captured_date, timestamp, embedding, app, title)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(timestamp) DO UPDATE SET
                    text=excluded.text,
                    captured_date=excluded.captured_date,
                    embedding=excluded.embedding,
                    app=excluded.app,
                    title=excluded.title
                """,
                (
                    text,
                    normalized_captured_date,
                    timestamp,
                    embedding_bytes,
                    app,
                    title,
                ),
            )
            conn.commit()
            last_row_id = cursor.lastrowid or None
    except sqlite3.Error as e:
        print(f"Database error during upsert: {e}")
    return last_row_id
