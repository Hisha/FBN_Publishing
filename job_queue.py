import os
import time
import uuid
import subprocess
import re
import shutil
import json
from datetime import datetime

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

    # Always generate internal filename
    internal_filename = f"{job_id}.png"

    # Sanitize optional custom filename
    requested_filename = params.get("filename")
    custom_filename = None
    if requested_filename:
        custom_filename = re.sub(r'[^a-zA-Z0-9_\-\.]', '', requested_filename)
        if not custom_filename.lower().endswith(".png"):
            custom_filename += ".png"

    # Handle output_dir if provided
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
        seed=params.get("seed")
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

def run_worker():
    while True:
        job = get_oldest_queued_job()
        if not job:
            time.sleep(1)
            continue

        job_id = job["job_id"]
        update_job_status(job_id, "in_progress", start_time=datetime.utcnow().isoformat())

        try:
            # Build command for run_flux.py
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
                "--quiet"  # ensure only JSON is printed
            ]

            # Apply flags based on job properties
            if job.get("autotune"):
                cmd.append("--autotune")
            if job.get("adults"):
                cmd.append("--adults")
            if job.get("cover_mode"):
                cmd.append("--cover_mode")
            if job.get("seed"):
                cmd.extend(["--seed", str(job["seed"])])

            print(f"▶ Running command: {' '.join(cmd)}")

            # Execute run_flux.py and capture output
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

            # ✅ Extract JSON from stdout (ignore other logs if present)
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
                    error_message=f"Invalid JSON output from run_flux.py: {str(e)}"
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

            # Capture key details
            final_path = result.get("file")
            final_filename = os.path.basename(final_path)
            upscaled = result.get("upscaled", False)
            mode = "cover" if job.get("cover_mode") else "coloring"

            # Update DB
            update_job_status(
                job_id,
                "done",
                end_time=datetime.utcnow().isoformat(),
                filename=final_filename,
                upscaled=upscaled,
                mode=mode
            )

            # Copy to custom output_dir if specified
            try:
                dest_dir = job.get("output_dir")
                if dest_dir:
                    os.makedirs(dest_dir, exist_ok=True)
                    custom_filename = job.get("custom_filename")
                    if custom_filename:
                        dest_path = os.path.join(dest_dir, custom_filename)
                    else:
                        dest_path = os.path.join(dest_dir, final_filename)
                    shutil.copy2(final_path, dest_path)
                    print(f"✅ Copied file to: {dest_path}")
            except Exception as copy_err:
                print(f"⚠️ Failed to copy to output_dir: {copy_err}")

        except Exception as e:
            update_job_status(job_id, "failed", end_time=datetime.utcnow().isoformat(), error_message=str(e))

# Initialize DB at startup
init_db()
print("✅ Job queue initialized.")
