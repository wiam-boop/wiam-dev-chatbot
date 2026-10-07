import os
import json
import sqlite3
from contextlib import contextmanager


DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()


# =========================================================
# DATABASE URL
# =========================================================

# Railway/PostgreSQL قد يستخدم أحيانًا postgres://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace(
        "postgres://",
        "postgresql://",
        1
    )


USE_POSTGRES = bool(DATABASE_URL)


# =========================================================
# SQLITE PATH
# =========================================================

# إذا كان STORAGE_PATH موجودًا، نستخدمه لتخزين SQLite
# بدل المسار الحالي.
_env_storage_path = os.environ.get("STORAGE_PATH", "").strip()

if _env_storage_path:
    os.makedirs(_env_storage_path, exist_ok=True)
    SQLITE_PATH = os.path.join(
        _env_storage_path,
        "chat_logs.db"
    )
else:
    SQLITE_PATH = "chat_logs.db"


# =========================================================
# SQLITE CONNECTION
# =========================================================

def _get_sqlite_connection():

    conn = sqlite3.connect(
        SQLITE_PATH,
        timeout=30,
        check_same_thread=False
    )

    # يجعل SQLite rows تتصرف مثل dictionaries
    conn.row_factory = sqlite3.Row

    return conn


# =========================================================
# POSTGRESQL CONNECTION
# =========================================================

def _get_postgres_connection():

    try:
        import psycopg
        from psycopg.rows import dict_row

    except ImportError as exc:

        raise RuntimeError(
            "مكتبة psycopg غير مثبتة. "
            "أضف psycopg[binary] إلى requirements.txt"
        ) from exc

    # dict_row مهم جدًا لأن الكود يستخدم dict(row)
    return psycopg.connect(
        DATABASE_URL,
        row_factory=dict_row
    )


# =========================================================
# GENERAL CONNECTION
# =========================================================

@contextmanager
def get_connection():

    conn = (
        _get_postgres_connection()
        if USE_POSTGRES
        else _get_sqlite_connection()
    )

    try:

        yield conn

        conn.commit()

    except Exception:

        conn.rollback()

        raise

    finally:

        conn.close()


# =========================================================
# CREATE DATABASE
# =========================================================

def init_db():

    with get_connection() as conn:

        cur = conn.cursor()

        # -------------------------------------------------
        # POSTGRESQL
        # -------------------------------------------------

        if USE_POSTGRES:

            cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id BIGSERIAL PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    image_url TEXT,
                    sources TEXT,
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_chat_messages_session
                ON chat_messages(session_id)
            """)

            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_chat_messages_created
                ON chat_messages(created_at)
            """)

        # -------------------------------------------------
        # SQLITE
        # -------------------------------------------------

        else:

            cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    image_url TEXT,
                    sources TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_chat_messages_session
                ON chat_messages(session_id)
            """)

            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_chat_messages_created
                ON chat_messages(created_at)
            """)


# =========================================================
# SAVE MESSAGE
# =========================================================

def save_message(
    session_id,
    role,
    content,
    image_url=None,
    sources=None
):

    if not session_id:
        return

    sources_json = json.dumps(
        sources or [],
        ensure_ascii=False
    )

    with get_connection() as conn:

        cur = conn.cursor()

        # -------------------------------------------------
        # POSTGRESQL
        # -------------------------------------------------

        if USE_POSTGRES:

            cur.execute(
                """
                INSERT INTO chat_messages
                (
                    session_id,
                    role,
                    content,
                    image_url,
                    sources
                )
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    session_id,
                    role,
                    content,
                    image_url,
                    sources_json
                )
            )

        # -------------------------------------------------
        # SQLITE
        # -------------------------------------------------

        else:

            cur.execute(
                """
                INSERT INTO chat_messages
                (
                    session_id,
                    role,
                    content,
                    image_url,
                    sources
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    role,
                    content,
                    image_url,
                    sources_json
                )
            )


# =========================================================
# GET ALL CONVERSATIONS
# =========================================================

def get_conversations(limit=100):

    with get_connection() as conn:

        cur = conn.cursor()

        # -------------------------------------------------
        # POSTGRESQL
        # -------------------------------------------------

        if USE_POSTGRES:

            cur.execute(
                """
                SELECT
                    session_id,
                    COUNT(*) AS message_count,
                    MIN(created_at) AS first_message,
                    MAX(created_at) AS last_message,
                    (
                        SELECT content
                        FROM chat_messages m2
                        WHERE m2.session_id = m1.session_id
                          AND m2.role = 'user'
                        ORDER BY m2.created_at ASC, m2.id ASC
                        LIMIT 1
                    ) AS first_user_message
                FROM chat_messages m1
                GROUP BY session_id
                ORDER BY MAX(created_at) DESC
                LIMIT %s
                """,
                (limit,)
            )

        # -------------------------------------------------
        # SQLITE
        # -------------------------------------------------

        else:

            cur.execute(
                """
                SELECT
                    session_id,
                    COUNT(*) AS message_count,
                    MIN(created_at) AS first_message,
                    MAX(created_at) AS last_message,
                    (
                        SELECT content
                        FROM chat_messages m2
                        WHERE m2.session_id = m1.session_id
                          AND m2.role = 'user'
                        ORDER BY m2.created_at ASC, m2.id ASC
                        LIMIT 1
                    ) AS first_user_message
                FROM chat_messages m1
                GROUP BY session_id
                ORDER BY MAX(created_at) DESC
                LIMIT ?
                """,
                (limit,)
            )

        return [
            dict(row)
            for row in cur.fetchall()
        ]


# =========================================================
# GET CONVERSATION MESSAGES
# =========================================================

def get_messages(
    session_id,
    limit=500
):

    with get_connection() as conn:

        cur = conn.cursor()

        # -------------------------------------------------
        # POSTGRESQL
        # -------------------------------------------------

        if USE_POSTGRES:

            cur.execute(
                """
                SELECT
                    id,
                    session_id,
                    role,
                    content,
                    image_url,
                    sources,
                    created_at
                FROM chat_messages
                WHERE session_id = %s
                ORDER BY created_at ASC, id ASC
                LIMIT %s
                """,
                (
                    session_id,
                    limit
                )
            )

        # -------------------------------------------------
        # SQLITE
        # -------------------------------------------------

        else:

            cur.execute(
                """
                SELECT
                    id,
                    session_id,
                    role,
                    content,
                    image_url,
                    sources,
                    created_at
                FROM chat_messages
                WHERE session_id = ?
                ORDER BY created_at ASC, id ASC
                LIMIT ?
                """,
                (
                    session_id,
                    limit
                )
            )

        rows = []

        for row in cur.fetchall():

            item = dict(row)

            try:

                item["sources"] = json.loads(
                    item.get("sources") or "[]"
                )

            except Exception:

                item["sources"] = []

            rows.append(item)

        return rows


# =========================================================
# DELETE CONVERSATION
# =========================================================

def delete_conversation(session_id):

    with get_connection() as conn:

        cur = conn.cursor()

        # -------------------------------------------------
        # POSTGRESQL
        # -------------------------------------------------

        if USE_POSTGRES:

            cur.execute(
                """
                DELETE FROM chat_messages
                WHERE session_id = %s
                """,
                (session_id,)
            )

            deleted = cur.rowcount > 0

        # -------------------------------------------------
        # SQLITE
        # -------------------------------------------------

        else:

            cur.execute(
                """
                DELETE FROM chat_messages
                WHERE session_id = ?
                """,
                (session_id,)
            )

            deleted = cur.rowcount > 0

        return deleted


# =========================================================
# STATISTICS
# =========================================================

def get_stats():

    with get_connection() as conn:

        cur = conn.cursor()

        # -------------------------------------------------
        # TOTAL MESSAGES
        # -------------------------------------------------

        cur.execute("""
            SELECT COUNT(*) AS total_messages
            FROM chat_messages
        """)

        row = cur.fetchone()

        if USE_POSTGRES:
            total_messages = row["total_messages"]
        else:
            total_messages = row[0]

        # -------------------------------------------------
        # TOTAL CONVERSATIONS
        # -------------------------------------------------

        cur.execute("""
            SELECT COUNT(DISTINCT session_id) AS total_conversations
            FROM chat_messages
        """)

        row = cur.fetchone()

        if USE_POSTGRES:
            total_conversations = row["total_conversations"]
        else:
            total_conversations = row[0]

        # -------------------------------------------------
        # TOTAL IMAGES
        # -------------------------------------------------

        cur.execute("""
            SELECT COUNT(*) AS total_images
            FROM chat_messages
            WHERE image_url IS NOT NULL
              AND image_url != ''
        """)

        row = cur.fetchone()

        if USE_POSTGRES:
            total_images = row["total_images"]
        else:
            total_images = row[0]

        # -------------------------------------------------
        # RETURN STATS
        # -------------------------------------------------

        return {
            "messages": total_messages,
            "conversations": total_conversations,
            "images": total_images
        }

