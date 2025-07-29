import sqlite3
import os
from datetime import datetime

DB_FILE = "jobs.db"

def get_conn():
    return sqlite3.connect(DB_FILE)

def init_db():
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS jobs (
        id TEXT PRIMARY KEY,
        prompt TEXT,
        adults INTEGER,
        upscale INTEGER,
        output_dir TEXT,
        filename TEXT,
        status TEXT,
        file_path TEXT,
        created_at TEXT,
        completed_at TEXT
    )
    """)
    conn.commit()
    conn.close()

def add_job(job_id, prompt, adults, upscale, output_dir=None, filename=None):
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
    INSERT INTO jobs (id, prompt, adults, upscale, output_dir, filename, status, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (job_id, prompt, int(adults), int(upscale), output_dir, filename, "queued", datetime.now().isoformat()))
    conn.commit()
    conn.close()

def get_jobs(status=None, limit=20):
    conn = get_conn()
    c = conn.cursor()
    if status:
        c.execute("SELECT * FROM jobs WHERE status=? ORDER BY created_at DESC LIMIT ?", (status, limit))
    else:
        c.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    return [row_to_dict(r) for r in rows]

def get_job_by_id(job_id):
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT * FROM jobs WHERE id=?", (job_id,))
    row = c.fetchone()
    conn.close()
    return row_to_dict(row) if row else None

def update_job_status(job_id, status, file_path=None):
    conn = get_conn()
    c = conn.cursor()
    if file_path:
        c.execute("UPDATE jobs SET status=?, file_path=?, completed_at=? WHERE id=?",
                  (status, file_path, datetime.now().isoformat(), job_id))
    else:
        c.execute("UPDATE jobs SET status=? WHERE id=?", (status, job_id))
    conn.commit()
    conn.close()

def row_to_dict(row):
    if not row:
        return None
    return {
        "id": row[0],
        "prompt": row[1],
        "adults": bool(row[2]),
        "upscale": bool(row[3]),
        "output_dir": row[4],
        "filename": row[5],
        "status": row[6],
        "file_path": row[7],
        "created_at": row[8],
        "completed_at": row[9]
    }
