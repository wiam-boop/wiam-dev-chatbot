import os
import json
import sqlite3
from contextlib import contextmanager

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
USE_POSTGRES = bool(DATABASE_URL)

_env_storage_path = os.environ.get("STORAGE_PATH", "").strip()
if _env_storage_path:
    os.makedirs(_env_storage_path, exist_ok=True)
    SQLITE_PATH = os.path.join(_env_storage_path, "chat_logs.db")
else:
    SQLITE_PATH = "chat_logs.db"

def _get_sqlite_connection():
    conn = sqlite3.connect(SQLITE_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def _get_postgres_connection():
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError("مكتبة psycopg غير مثبتة. أضف psycopg[binary] إلى requirements.txt") from exc
    return psycopg.connect(DATABASE_URL)

@contextmanager
def get_connection():
    conn = _get_postgres_connection() if USE_POSTGRES else _get_sqlite_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_db():
    with get_connection() as conn:
        cur = conn.cursor()
        if USE_POSTGRES:
            cur.execute("""CREATE TABLE IF NOT EXISTS chat_messages (
                id BIGSERIAL PRIMARY KEY, session_id TEXT NOT NULL, role TEXT NOT NULL,
                content TEXT NOT NULL, image_url TEXT, sources TEXT,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP)""")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_created ON chat_messages(created_at)")
            cur.execute("""CREATE TABLE IF NOT EXISTS learned_knowledge (
                id BIGSERIAL PRIMARY KEY, question TEXT NOT NULL, answer TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'user', source_urls TEXT,
                embedding DOUBLE PRECISION[] NOT NULL,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP)""")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_learned_knowledge_created ON learned_knowledge(created_at)")
        else:
            cur.execute("""CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, role TEXT NOT NULL,
                content TEXT NOT NULL, image_url TEXT, sources TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_created ON chat_messages(created_at)")
            cur.execute("""CREATE TABLE IF NOT EXISTS learned_knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT, question TEXT NOT NULL, answer TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'user', source_urls TEXT, embedding TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP, updated_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_learned_knowledge_created ON learned_knowledge(created_at)")

def save_message(session_id, role, content, image_url=None, sources=None):
    if not session_id:
        return
    sources_json = json.dumps(sources or [], ensure_ascii=False)
    with get_connection() as conn:
        cur = conn.cursor()
        if USE_POSTGRES:
            cur.execute("""INSERT INTO chat_messages (session_id, role, content, image_url, sources)
                VALUES (%s, %s, %s, %s, %s)""", (session_id, role, content, image_url, sources_json))
        else:
            cur.execute("""INSERT INTO chat_messages (session_id, role, content, image_url, sources)
                VALUES (?, ?, ?, ?, ?)""", (session_id, role, content, image_url, sources_json))

def save_learned_knowledge(question, answer, embedding, source="user", source_urls=None):
    question = str(question or "").strip()
    answer = str(answer or "").strip()
    if not question or not answer:
        raise ValueError("question و answer مطلوبان")
    source_urls_json = json.dumps(source_urls or [], ensure_ascii=False)
    embedding = [float(x) for x in embedding]
    with get_connection() as conn:
        cur = conn.cursor()
        if USE_POSTGRES:
            cur.execute("""INSERT INTO learned_knowledge
                (question, answer, source, source_urls, embedding)
                VALUES (%s, %s, %s, %s, %s) RETURNING id""",
                (question, answer, source, source_urls_json, embedding))
            row = cur.fetchone()
            return row[0] if row else None
        cur.execute("""INSERT INTO learned_knowledge
            (question, answer, source, source_urls, embedding) VALUES (?, ?, ?, ?, ?)""",
            (question, answer, source, source_urls_json, json.dumps(embedding)))
        return cur.lastrowid

def get_all_learned_knowledge():
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""SELECT id, question, answer, source, source_urls, embedding,
            created_at, updated_at FROM learned_knowledge ORDER BY created_at ASC, id ASC""")
        result = []
        for row in cur.fetchall():
            item = dict(row)
            emb = item.get("embedding")
            if isinstance(emb, str):
                try: emb = json.loads(emb)
                except Exception: emb = []
            item["embedding"] = emb or []
            try: item["source_urls"] = json.loads(item.get("source_urls") or "[]")
            except Exception: item["source_urls"] = []
            result.append(item)
        return result

def update_learned_knowledge(memory_id, question, answer, embedding, source="user", source_urls=None):
    source_urls_json = json.dumps(source_urls or [], ensure_ascii=False)
    embedding = [float(x) for x in embedding]
    with get_connection() as conn:
        cur = conn.cursor()
        if USE_POSTGRES:
            cur.execute("""UPDATE learned_knowledge SET question=%s, answer=%s, source=%s,
                source_urls=%s, embedding=%s, updated_at=CURRENT_TIMESTAMP WHERE id=%s""",
                (question, answer, source, source_urls_json, embedding, memory_id))
        else:
            cur.execute("""UPDATE learned_knowledge SET question=?, answer=?, source=?,
                source_urls=?, embedding=?, updated_at=CURRENT_TIMESTAMP WHERE id=?""",
                (question, answer, source, source_urls_json, json.dumps(embedding), memory_id))
        return cur.rowcount > 0

def delete_learned_knowledge(memory_id):
    with get_connection() as conn:
        cur = conn.cursor()
        if USE_POSTGRES:
            cur.execute("DELETE FROM learned_knowledge WHERE id=%s", (memory_id,))
        else:
            cur.execute("DELETE FROM learned_knowledge WHERE id=?", (memory_id,))
        return cur.rowcount > 0

def count_learned_knowledge():
    with get_connection() as conn:
        cur = conn.cursor(); cur.execute("SELECT COUNT(*) FROM learned_knowledge")
        return int(cur.fetchone()[0])

def get_conversations(limit=100):
    with get_connection() as conn:
        cur = conn.cursor()
        if USE_POSTGRES:
            cur.execute("""SELECT session_id, COUNT(*) AS message_count, MIN(created_at) AS first_message,
                MAX(created_at) AS last_message, (SELECT content FROM chat_messages m2
                WHERE m2.session_id=m1.session_id AND m2.role='user'
                ORDER BY m2.created_at ASC, m2.id ASC LIMIT 1) AS first_user_message
                FROM chat_messages m1 GROUP BY session_id ORDER BY MAX(created_at) DESC LIMIT %s""", (limit,))
        else:
            cur.execute("""SELECT session_id, COUNT(*) AS message_count, MIN(created_at) AS first_message,
                MAX(created_at) AS last_message, (SELECT content FROM chat_messages m2
                WHERE m2.session_id=m1.session_id AND m2.role='user'
                ORDER BY m2.created_at ASC, m2.id ASC LIMIT 1) AS first_user_message
                FROM chat_messages m1 GROUP BY session_id ORDER BY MAX(created_at) DESC LIMIT ?""", (limit,))
        return [dict(row) for row in cur.fetchall()]

def get_messages(session_id, limit=500):
    with get_connection() as conn:
        cur = conn.cursor()
        if USE_POSTGRES:
            cur.execute("""SELECT id, session_id, role, content, image_url, sources, created_at
                FROM chat_messages WHERE session_id=%s ORDER BY created_at ASC, id ASC LIMIT %s""", (session_id, limit))
        else:
            cur.execute("""SELECT id, session_id, role, content, image_url, sources, created_at
                FROM chat_messages WHERE session_id=? ORDER BY created_at ASC, id ASC LIMIT ?""", (session_id, limit))
        rows=[]
        for row in cur.fetchall():
            item=dict(row)
            try: item["sources"]=json.loads(item.get("sources") or "[]")
            except Exception: item["sources"]=[]
            rows.append(item)
        return rows

def delete_conversation(session_id):
    with get_connection() as conn:
        cur=conn.cursor()
        if USE_POSTGRES: cur.execute("DELETE FROM chat_messages WHERE session_id=%s", (session_id,))
        else: cur.execute("DELETE FROM chat_messages WHERE session_id=?", (session_id,))
        return cur.rowcount > 0

def get_stats():
    with get_connection() as conn:
        cur=conn.cursor()
        cur.execute("SELECT COUNT(*) FROM chat_messages"); total_messages=cur.fetchone()[0]
        cur.execute("SELECT COUNT(DISTINCT session_id) FROM chat_messages"); total_conversations=cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM chat_messages WHERE image_url IS NOT NULL AND image_url != ''"); total_images=cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM learned_knowledge"); total_learned=cur.fetchone()[0]
        return {"messages":total_messages,"conversations":total_conversations,"images":total_images,"learned_knowledge":total_learned}
