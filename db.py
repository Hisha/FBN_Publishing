import sqlite3
import os
from datetime import datetime
import pytz
from dateutil import parser

DB_PATH = os.path.expanduser("~/FBN_publishing/fbnp_jobs.db")
eastern = pytz.timezone("US/Eastern")


# ==========================
# ✅ Utility: Time Formatting
# ==========================
def format_local_time(iso_str):
    try:
        utc_time = parser.isoparse(iso_str)
        local_time = utc_time.astimezone(eastern)
        return local_time.strftime("%Y-%m-%d %I:%M %p %Z")  # e.g., 2025-07-15 02:30 PM EDT
    except Exception:
        return iso_str


# ==========================
# ✅ Initialize Database
# ==========================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS jobs (
        job_id TEXT PRIMARY KEY,
        prompt TEXT,
        steps INTEGER,
        guidance_scale REAL,
        height INTEGER,
        width INTEGER,
        autotune INTEGER,
        adults INTEGER,
        cover_mode INTEGER,
        upscaled INTEGER,
        mode TEXT,
        seed INTEGER,
        status TEXT,
        filename TEXT,
        custom_filename TEXT,
        output_dir TEXT,
        start_time TEXT,
        end_time TEXT,
        error_message TEXT
    )
    ''')
    conn.commit()
    conn.close()


# ==========================
# ✅ Add a Job
# ==========================
def add_job(job_id, prompt, steps, guidance_scale, height, width, autotune,
            adults, cover_mode, output_dir=None, custom_filename=None, seed=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
    INSERT INTO jobs (
        job_id, prompt, steps, guidance_scale, height, width, autotune,
        adults, cover_mode, upscaled, mode, seed, status, filename,
        custom_filename, output_dir
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, 'queued', NULL, ?, ?)
    ''', (job_id, prompt, steps, guidance_scale, height, width, int(autotune),
          int(adults), int(cover_mode), seed, custom_filename, output_dir))
    conn.commit()
    conn.close()

def count_jobs_by_status(status: str) -> int:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM jobs WHERE status = ?", (status,))
    count = c.fetchone()[0]
    conn.close()
    return count

# ==========================
# ✅ Update Job Status
# ==========================
def update_job_status(job_id, status, start_time=None, end_time=None,
                      error_message=None, filename=None, upscaled=None, mode=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    fields = ["status = ?"]
    values = [status]

    if start_time:
        fields.append("start_time = ?")
        values.append(start_time)
    if end_time:
        fields.append("end_time = ?")
        values.append(end_time)
    if error_message:
        fields.append("error_message = ?")
        values.append(error_message)
    if filename:
        fields.append("filename = ?")
        values.append(filename)
    if upscaled is not None:
        fields.append("upscaled = ?")
        values.append(int(upscaled))
    if mode:
        fields.append("mode = ?")
        values.append(mode)

    values.append(job_id)
    c.execute(f'''
    UPDATE jobs SET {', '.join(fields)} WHERE job_id = ?
    ''', values)
    conn.commit()
    conn.close()


# ==========================
# ✅ Getters
# ==========================
def get_job(job_id):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM jobs WHERE job_id = ?', (job_id,))
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None

def get_job_by_filename(filename):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM jobs WHERE filename = ?", (filename,))
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None

def get_all_jobs():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM jobs ORDER BY start_time DESC")
    rows = c.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_completed_jobs_for_archive(days=1):
    cutoff = datetime.utcnow().timestamp() - (days * 86400)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('''
        SELECT job_id, filename, end_time FROM jobs
        WHERE status = 'done' AND COALESCE(strftime('%s', end_time), 0) < ?
    ''', (cutoff,))
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_recent_jobs(limit=50, status=None):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    query = "SELECT * FROM jobs"
    values = []
    if status and status != "all":
        query += " WHERE status = ?"
        values.append(status)

    c.execute(query, values)
    rows = c.fetchall()
    conn.close()

    status_priority = {
        "in_progress": 0,
        "queued": 1,
        "failed": 2,
        "done": 3
    }

    def sort_key(job):
        prio = status_priority.get(job["status"], 99)
        raw_ts = job.get("end_time") or job.get("start_time")
        try:
            ts = datetime.fromisoformat(raw_ts).timestamp()
        except:
            ts = 0
        return (prio, -ts)

    return sorted([dict(r) for r in rows], key=sort_key)[:limit]

def get_job_metrics():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute("SELECT COUNT(*) as total FROM jobs")
    total = c.fetchone()["total"]

    c.execute("SELECT COUNT(*) as completed FROM jobs WHERE status = 'done'")
    done = c.fetchone()["completed"]

    c.execute("SELECT COUNT(*) as failed FROM jobs WHERE status = 'failed'")
    failed = c.fetchone()["failed"]

    # Duration in seconds
    c.execute('''
        SELECT AVG(strftime('%s', end_time) - strftime('%s', start_time)) as avg_duration
        FROM jobs
        WHERE status = 'done' AND start_time IS NOT NULL AND end_time IS NOT NULL
    ''')
    duration = c.fetchone()["avg_duration"]

    c.execute("SELECT MAX(start_time) as last_job FROM jobs")
    last_job = c.fetchone()["last_job"]

    conn.close()
    return {
        "total_jobs": total,
        "completed_jobs": done,
        "failed_jobs": failed,
        "average_duration_seconds": round(duration or 0, 2),
        "most_recent_job_time": format_local_time(last_job) if last_job else "N/A"
    }

def get_oldest_queued_job():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("BEGIN IMMEDIATE")
    c.execute("""
        SELECT * FROM jobs
        WHERE status = 'queued'
        ORDER BY rowid ASC
        LIMIT 1
    """)
    row = c.fetchone()
    if row:
        job_id = row["job_id"]
        now = datetime.utcnow().isoformat()
        c.execute('''
            UPDATE jobs
            SET status = 'in_progress',
                start_time = ?
            WHERE job_id = ?
        ''', (now, job_id))
        conn.commit()
        conn.close()
        return dict(row)
    else:
        conn.commit()
        conn.close()
        return None


# ==========================
# ✅ Delete or Cleanup
# ==========================
def delete_job(job_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT filename FROM jobs WHERE job_id = ?", (job_id,))
    row = c.fetchone()
    filename = row[0] if row else None
    c.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
    conn.commit()
    conn.close()
    return filename

def delete_old_jobs(days=7):
    cutoff = datetime.utcnow().timestamp() - (days * 86400)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Get jobs to delete
    c.execute('''
        SELECT job_id, filename FROM jobs
        WHERE 
            status IN ('done', 'failed') AND
            COALESCE(strftime('%s', end_time), 0) < ?
    ''', (cutoff,))
    jobs = c.fetchall()

    # Delete them
    c.execute('''
        DELETE FROM jobs
        WHERE 
            status IN ('done', 'failed') AND
            COALESCE(strftime('%s', end_time), 0) < ?
    ''', (cutoff,))
    conn.commit()
    conn.close()

    return [dict(job) for job in jobs]

def delete_queued_jobs():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM jobs WHERE status = 'queued'")
    conn.commit()
    conn.close()


def get_job_metrics():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute("SELECT COUNT(*) as total FROM jobs")
    total = c.fetchone()["total"]

    c.execute("SELECT COUNT(*) as completed FROM jobs WHERE status = 'done'")
    done = c.fetchone()["completed"]

    c.execute("SELECT COUNT(*) as failed FROM jobs WHERE status = 'failed'")
    failed = c.fetchone()["failed"]

    c.execute('''
        SELECT AVG(strftime('%s', end_time) - strftime('%s', start_time)) as avg_duration
        FROM jobs
        WHERE status = 'done' AND start_time IS NOT NULL AND end_time IS NOT NULL
    ''')
    duration = c.fetchone()["avg_duration"]

    c.execute("SELECT MAX(start_time) as last_job FROM jobs")
    last_job = c.fetchone()["last_job"]

    conn.close()
    return {
        "total_jobs": total,
        "completed_jobs": done,
        "failed_jobs": failed,
        "average_duration_seconds": round(duration or 0, 2),
        "most_recent_job_time": format_local_time(last_job) if last_job else "N/A"
    }
