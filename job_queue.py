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

    # Internal filename for tracking
    internal_filename = f"{job_id}.png"

    # Sanitize custom filename if provided
    requested_filename = params.get("filename")
    custom_filename = None
    if requested_filename:
        custom_filename = re.sub(r'[^a-zA-Z0-9_\-\.]', '', requested_filename)
        if not custom_filename.lower().endswith(".png"):
            custom_filename += ".png"

    # Handle output directory if provided
    output_dir = params.get("output_dir")
    if output_dir:
        output_dir = os.path.abspath(os.path.expanduser(output_dir))
        os.makedirs(output_dir, exist_ok=True)

    # Insert into DB
    add_job(
        job_id=job_id,
        prompt=params["prompt"],
        steps=params.get("steps", 4),
        guidance_scale=params.get("guidance_scale", 3.5),
        height=params.get("height", 1088),
        width=params.get("width", 848),
        autotune=params.get("autotune", True),
        adults=params.get("adults", False),
        cover_mode=params.get("cover_mode", False),
        output_dir=output_dir,
        custom_filename=custom_filename,
        seed=params.get("seed"),
        page_count=job.get("page_count")
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
            # Build run_flux.py command
            cmd = [
                PYTHON_BIN,
                RUN_FLUX_SCRIPT,
                "--prompt", job["prompt"],
                "--output", f"{job_id}.png",
                "--output_dir", OUTPUT_DIR,
                "--steps", str(job["steps"]),
                "--guidance_scale", str(job["guidance_scale"]),
                "--height", str(job["height"]),
                "--width", str(job["width"]),
                "--quiet"
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

            # Execute and capture output
            process = subprocess.run(cmd, capture_output=True, text=True)
            stdout, stderr = process.stdout.strip(), process.stderr.strip()

            if process.returncode != 0:
                update_job_status(
                    job_id,
                    "failed",
                    end_time=datetime.utcnow().isoformat(),
                    error_message=stderr or "Unknown error"
                )
                print(f"❌ Process failed: {stderr}")
                continue

            # Extract JSON result
            try:
                json_match = re.search(r"\{.*\}", stdout, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    raise ValueError("No JSON found in output")
            except Exception as e:
                update_job_status(
                    job_id,
                    "failed",
                    end_time=datetime.utcnow().isoformat(),
                    error_message=f"Invalid JSON output: {str(e)}"
                )
                print(f"⚠️ STDOUT:\n{stdout}")
                print(f"⚠️ STDERR:\n{stderr}")
                continue

            if result.get("status") != "success":
                update_job_status(
                    job_id,
                    "failed",
                    end_time=datetime.utcnow().isoformat(),
                    error_message=result.get("error", "Unknown error from generator")
                )
                continue

            final_path = result.get("file")
            upscaled = result.get("upscaled", False)

            # ✅ If upscaled, remove original and rename upscaled to original name
            if upscaled and "_upscaled" in final_path:
                original_path = os.path.join(OUTPUT_DIR, f"{job_id}.png")
                if os.path.exists(original_path):
                    os.remove(original_path)
                new_final_path = original_path
                shutil.move(final_path, new_final_path)
                final_path = new_final_path

            final_filename = os.path.basename(final_path)
            mode = "cover" if job.get("cover_mode") else "coloring"

            # ✅ Get actual resolution using ffprobe
            try:
                ffprobe_cmd = [
                    "ffprobe", "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries", "stream=width,height",
                    "-of", "csv=s=x:p=0", final_path
                ]
                resolution = subprocess.check_output(ffprobe_cmd, text=True).strip()
                width, height = map(int, resolution.split('x'))
            except Exception as e:
                print(f"⚠️ Could not get resolution: {e}")
                width, height = job["width"], job["height"]

            # ✅ Update DB with correct resolution
            update_job_status(
                job_id,
                "done",
                end_time=datetime.utcnow().isoformat(),
                filename=final_filename,
                upscaled=upscaled,
                mode=mode,
                width=width,
                height=height
            )

            # ✅ Copy to custom output_dir if provided
            try:
                dest_dir = job.get("output_dir")
                if dest_dir:
                    os.makedirs(dest_dir, exist_ok=True)
                    custom_filename = job.get("custom_filename")
                    dest_path = os.path.join(dest_dir, custom_filename or final_filename)
                    shutil.copy2(final_path, dest_path)
                    print(f"✅ Copied file to: {dest_path}")
            except Exception as copy_err:
                print(f"⚠️ Failed to copy to output_dir: {copy_err}")

            # ✅ Create thumbnail for gallery
            thumb_dir = os.path.join(OUTPUT_DIR, "thumbnails")
            os.makedirs(thumb_dir, exist_ok=True)
            final_filename = os.path.basename(final_output_path)
            thumb_path = os.path.join(thumb_dir, final_filename)
            create_thumbnail(final_output_path, thumb_path)
            print(f"✅ Thumbnail created at {thumb_path}")

        except Exception as e:
            update_job_status(job_id, "failed", end_time=datetime.utcnow().isoformat(), error_message=str(e))


# Init DB
init_db()
print("✅ Job queue initialized.")
