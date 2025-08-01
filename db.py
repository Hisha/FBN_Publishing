import sqlite3
import os
from datetime import datetime
import pytz
from dateutil import parser

DB_PATH = os.path.expanduser("~/FBN_publishing/fbnp_jobs.db")
eastern = pytz.timezone("US/Eastern")


def format_local_time(iso_str):
    try:
        utc_time = parser.isoparse(iso_str)
        local_time = utc_time.astimezone(eastern)
        return local_time.strftime("%Y-%m-%d %I:%M %p %Z")
    except Exception:
        return iso_str


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
        error_message TEXT,
        page_count INTEGER
    )
    ''')
    conn.commit()
    conn.close()


# ✅ Modified: height & width now optional (default None)
def add_job(job_id, prompt, steps, guidance_scale, autotune,
            adults, cover_mode, output_dir=None, custom_filename=None, seed=None, page_count=None,
            height=None, width=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
    INSERT INTO jobs (
        job_id, prompt, steps, guidance_scale, height, width, autotune,
        adults, cover_mode, upscaled, mode, seed, page_count, status, filename,
        custom_filename, output_dir
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, 'queued', NULL, ?, ?)
    ''', (job_id, prompt, steps, guidance_scale, height, width, int(autotune),
          int(adults), int(cover_mode), seed, page_count, custom_filename, output_dir))
    conn.commit()
    conn.close()


def count_jobs_by_status(status: str) -> int:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM jobs WHERE status = ?", (status,))
    count = c.fetchone()[0]
    conn.close()
    return count


def update_job_status(job_id, status, start_time=None, end_time=None,
                      error_message=None, filename=None, upscaled=None, mode=None,
                      width=None, height=None):
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
    if width:
        fields.append("width = ?")
        values.append(width)
    if height:
        fields.append("height = ?")
        values.append(height)

    values.append(job_id)
    c.execute(f'''
    UPDATE jobs SET {', '.join(fields)} WHERE job_id = ?
    ''', values)
    conn.commit()
    conn.close()
