import os
import time
import uuid
import subprocess
import re
import shutil
import json
from datetime import datetime
from PIL import Image

from db import (
    add_job,
    update_job_status,
    init_db,
    get_oldest_queued_job,
    delete_queued_jobs
)

OUTPUT_DIR = os.path.expanduser("~/FluxImages")
PYTHON_BIN = "/home/smithkt/FBN_publishing/FBNP_env/bin/python"
RUN_FLUX_SCRIPT = "/home/smithkt/FBN_publishing/run_flux.py"


def add_job_to_db_and_queue(params):
    job_id = uuid.uuid4().hex[:8]

    # Internal filename
    internal_filename = f"{job_id}.png"

    # Sanitize custom filename
    requested_filename = params.get("filename")
    custom_filename = None
    if requested_filename:
        custom_filename = re.sub(r'[^a-zA-Z0-9_\-\.]', '', requested_filename)
        if not custom_filename.lower().endswith(".png"):
            custom_filename += ".png"

    # Output directory
    output_dir = params.get("output_dir")
    if output_dir:
        output_dir = os.path.abspath(os.path.expanduser(output_dir))
        os.makedirs(output_dir, exist_ok=True)

    # ✅ Insert into DB (height/width now None)
    add_job(
        job_id=job_id,
        prompt=params["prompt"],
        steps=params.get("steps", 4),
        guidance_scale=params.get("guidance_scale", 3.5),
        autotune=params.get("autotune", True),
        adults=params.get("adults", False),
        cover_mode=params.get("cover_mode", False),
        filename=internal_filename,
        output_dir=output_dir,
        custom_filename=custom_filename,
        seed=params.get("seed"),
        page_count=params.get("page_count"),
        height=None,
        width=None
    )

    return {
        "job_id": job_id,
        "status": "queued",
        "filename": internal_filename,
        "output_dir": output_dir,
        "custom_filename": custom_filename
    }


def clear_queue():
    delete_queued_jobs()


def create_thumbnail(source_path, dest_path, size=(400, 400)):
    try:
        img = Image.open(source_path)
        img.thumbnail(size)
        img.save(dest_path, "PNG", optimize=True)
        return True
    except Exception as e:
        print(f"⚠️ Failed to create thumbnail: {e}")
        return False

def run_worker():
    while True:
        job = get_oldest_queued_job()
        if not job:
            time.sleep(1)
            continue

        job_id = job["job_id"]
        update_job_status(job_id, "in_progress", start_time=datetime.utcnow().isoformat())

        try:
            # ✅ Build command
            cmd = [
                PYTHON_BIN,
                RUN_FLUX_SCRIPT,
                "--prompt", job["prompt"],
                "--output", f"{job_id}.png",
                "--output_dir", OUTPUT_DIR,
                "--steps", str(job["steps"]),
                "--guidance_scale", str(job["guidance_scale"])
            ]

            if job.get("autotune"):
                cmd.append("--autotune")
            if job.get("adults"):
                cmd.append("--adults")
            if job.get("cover_mode"):
                cmd.append("--cover_mode")
            if job.get("seed"):
                cmd.extend(["--seed", str(job["seed"])])
            if job.get("page_count"):
                cmd.extend(["--page_count", str(job["page_count"])])

            print(f"▶ Running command: {' '.join(cmd)}")

            # ✅ Run and capture JSON
            process = subprocess.run(cmd, capture_output=True, text=True)
            stdout, stderr = process.stdout.strip(), process.stderr.strip()

            if process.returncode != 0:
                update_job_status(job_id, "failed", end_time=datetime.utcnow().isoformat(),
                                  error_message=stderr or "Unknown error")
                print(f"❌ Process failed: {stderr}")
                continue

            # ✅ Parse JSON output from run_flux.py
            final_path, upscaled, width, height = None, False, None, None
            try:
                json_match = re.search(r"\{.*\}", stdout, re.DOTALL)
                if not json_match:
                    raise ValueError("No JSON block in output")

                result = json.loads(json_match.group())

                if result.get("status") != "success":
                    raise ValueError(f"Job error: {result.get('error', 'Unknown error')}")

                final_path = result.get("file")
                upscaled = result.get("upscaled", False)
                width = result.get("width")
                height = result.get("height")

            except Exception as e:
                print(f"⚠️ JSON parse failed, falling back: {e}")

            # ✅ Fallback if JSON didn't give width/height or file path
            if not final_path:
                raise ValueError("No file path found in JSON or fallback")
            if not width or not height:
                try:
                    ffprobe_cmd = [
                        "ffprobe", "-v", "error",
                        "-select_streams", "v:0",
                        "-show_entries", "stream=width,height",
                        "-of", "csv=s=x:p=0", final_path
                    ]
                    resolution = subprocess.check_output(ffprobe_cmd, text=True).strip()
                    width, height = map(int, resolution.split('x'))
                except Exception:
                    print("⚠️ Failed to get resolution via ffprobe")

            final_filename = os.path.basename(final_path)

            # ✅ Update DB
            update_job_status(job_id, "done",
                              end_time=datetime.utcnow().isoformat(),
                              filename=final_filename,
                              upscaled=upscaled,
                              width=width,
                              height=height)

            print(f"✅ Job {job_id} complete: {width}x{height}")

            # ✅ Copy to output_dir if provided
            dest_dir = job.get("output_dir")
            if dest_dir:
                os.makedirs(dest_dir, exist_ok=True)
                custom_filename = job.get("custom_filename")
                dest_path = os.path.join(dest_dir, custom_filename or final_filename)
                shutil.copy2(final_path, dest_path)
                print(f"✅ Copied file to: {dest_path}")

            # ✅ Create thumbnail
            thumb_dir = os.path.join(OUTPUT_DIR, "thumbnails")
            os.makedirs(thumb_dir, exist_ok=True)
            thumb_path = os.path.join(thumb_dir, final_filename)
            create_thumbnail(final_path, thumb_path)
            print(f"✅ Thumbnail created at {thumb_path}")

        except Exception as e:
            update_job_status(job_id, "failed",
                              end_time=datetime.utcnow().isoformat(),
                              error_message=str(e))

init_db()
print("✅ Job queue initialized.")
