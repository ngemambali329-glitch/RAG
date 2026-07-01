"""
chat_history.py
-----------------
Persistent chat history storage using a local SQLite database. Lets the app
show a list of previous chat sessions (like a sidebar in a typical chat app)
and reload past conversations after restarting the app.

This is intentionally simple (no external DB server needed) — the whole
history lives in one file: chat_history.db (excluded from git via .gitignore).
"""
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import config

DB_PATH = config.BASE_DIR / "chat_history.db"


@contextmanager
def _connect():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions (id)
            )
            """
        )


def create_session(first_message: str) -> str:
    """Create a new chat session, titled from the first question, and return its id."""
    session_id = str(uuid.uuid4())
    title = (first_message.strip()[:60] + "...") if len(first_message) > 60 else first_message.strip()
    now = datetime.utcnow().isoformat()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (session_id, title or "New conversation", now, now),
        )
    return session_id


def add_message(session_id: str, role: str, content: str, sources: Optional[List[Dict]] = None):
    now = datetime.utcnow().isoformat()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, sources_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, role, content, json.dumps(sources or []), now),
        )
        conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))


def list_sessions(limit: int = 50) -> List[Dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, title, created_at, updated_at FROM sessions "
            "ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_messages(session_id: str) -> List[Dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT role, content, sources_json, created_at FROM messages "
            "WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
    messages = []
    for row in rows:
        messages.append(
            {
                "role": row["role"],
                "content": row["content"],
                "sources": json.loads(row["sources_json"] or "[]"),
                "created_at": row["created_at"],
            }
        )
    return messages


def delete_session(session_id: str):
    with _connect() as conn:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))


def rename_session(session_id: str, new_title: str):
    with _connect() as conn:
        conn.execute("UPDATE sessions SET title = ? WHERE id = ?", (new_title, session_id))


init_db()
