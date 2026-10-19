import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path


class BusyError(RuntimeError):
    pass


class Store:
    def __init__(self, path):
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    user_id TEXT NOT NULL, session_id TEXT NOT NULL,
                    summary TEXT NOT NULL DEFAULT '', history TEXT NOT NULL DEFAULT '[]',
                    owner TEXT, lease_until REAL NOT NULL DEFAULT 0,
                    PRIMARY KEY(user_id, session_id)
                );
                CREATE TABLE IF NOT EXISTS todos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL, session_id TEXT NOT NULL,
                    title TEXT NOT NULL, done INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS todos_scope ON todos(user_id, session_id);
                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL, session_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL, ts REAL NOT NULL,
                    kind TEXT NOT NULL, data TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS events_scope ON events(user_id, session_id, seq);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def acquire(self, user, session, owner):
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO sessions(user_id, session_id) VALUES (?,?)", (user, session))
            changed = db.execute(
                "UPDATE sessions SET owner=?, lease_until=? WHERE user_id=? AND session_id=? "
                "AND (owner IS NULL OR lease_until < ?)",
                (owner, time.time() + 300, user, session, time.time()),
            ).rowcount
            if not changed:
                raise BusyError("Session is busy; retry after the current turn completes.")

    def renew(self, user, session, owner):
        with self.connect() as db:
            changed = db.execute(
                "UPDATE sessions SET lease_until=? WHERE user_id=? AND session_id=? "
                "AND owner=? AND lease_until >= ?",
                (time.time() + 300, user, session, owner, time.time()),
            ).rowcount
            if not changed:
                raise BusyError("Session lease expired or was replaced.")

    def release(self, user, session, owner):
        with self.connect() as db:
            db.execute("UPDATE sessions SET owner=NULL, lease_until=0 WHERE user_id=? AND session_id=? AND owner=?",
                       (user, session, owner))

    def load(self, user, session):
        with self.connect() as db:
            row = db.execute("SELECT summary, history FROM sessions WHERE user_id=? AND session_id=?",
                             (user, session)).fetchone()
        return (row["summary"], json.loads(row["history"])) if row else ("", [])

    def save(self, user, session, owner, summary, history):
        with self.connect() as db:
            changed = db.execute(
                "UPDATE sessions SET summary=?, history=? WHERE user_id=? AND session_id=? AND owner=? AND lease_until>=?",
                (summary, json.dumps(history, ensure_ascii=False), user, session, owner, time.time()),
            ).rowcount
            if not changed:
                raise BusyError("Cannot save a session without its active lease.")

    def todo(self, user, session, action, title=None, item_id=None):
        with self.connect() as db:
            if action == "add":
                count = db.execute("SELECT COUNT(*) FROM todos WHERE user_id=? AND session_id=?",
                                   (user, session)).fetchone()[0]
                if count >= 50:
                    raise ValueError("At most 50 todos per session; delete old items first.")
                if not title or not title.strip():
                    raise ValueError("title is required for add")
                item_id = db.execute("INSERT INTO todos(user_id, session_id, title) VALUES (?,?,?)",
                                     (user, session, title.strip())).lastrowid
            elif action in ("complete", "delete"):
                sql = ("UPDATE todos SET done=1 WHERE id=? AND user_id=? AND session_id=?"
                       if action == "complete" else "DELETE FROM todos WHERE id=? AND user_id=? AND session_id=?")
                if not db.execute(sql, (item_id, user, session)).rowcount:
                    raise ValueError("Todo not found in this session")
            elif action != "list":
                raise ValueError("Unknown todo action")
            items = [dict(row) for row in db.execute(
                "SELECT id, title, done FROM todos WHERE user_id=? AND session_id=? ORDER BY id", (user, session))]
        return {"items": items, "affected_id": item_id}

    def event(self, user, session, turn, kind, data):
        with self.connect() as db:
            db.execute("INSERT INTO events(user_id, session_id, turn_id, ts, kind, data) VALUES (?,?,?,?,?,?)",
                       (user, session, turn, time.time(), kind, json.dumps(data, ensure_ascii=False)))

    def events(self, user, session, limit=100):
        with self.connect() as db:
            rows = db.execute("SELECT * FROM (SELECT seq, turn_id, ts, kind, data FROM events "
                              "WHERE user_id=? AND session_id=? ORDER BY seq DESC LIMIT ?) ORDER BY seq",
                              (user, session, limit)).fetchall()
        return [{**dict(row), "data": json.loads(row["data"])} for row in rows]
