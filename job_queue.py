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

ENV = os.environ.copy()
ENV["PYTHONUNBUFFERED"] = "1"
OUTPUT_DIR = os.path.expanduser("~/FluxImages")
PYTHON_BIN = "/home/smithkt/FBN_publishing/FBNP_env/bin/python"
RUN_FLUX_SCRIPT = "/home/smithkt/FBN_publishing/run_flux.py"

# ⬇️ NEW: grayscale guard configuration (env-tunable)
ENFORCE_GRAYSCALE = os.getenv("ENFORCE_GRAYSCALE", "1") == "1"
MAX_COLOR_RETRIES = int(os.getenv("MAX_COLOR_RETRIES", "2"))
COLOR_SAT_THRESH = int(os.getenv("COLOR_SAT_THRESH", "18"))      # 0-255; ~18≈7% sat
COLOR_FRACTION   = float(os.getenv("COLOR_FRACTION", "0.003"))    # 0.3% of pixels
CROP_BORDER_PX   = int(os.getenv("COLOR_CROP_BORDER_PX", "4"))    # ignore thin borders

def _is_color_image(path: str,
                    sat_thresh: int = COLOR_SAT_THRESH,
                    color_frac: float = COLOR_FRACTION,
                    crop_border: int = CROP_BORDER_PX) -> bool:
    """
    Returns True if image has meaningful color content.
    Heuristic: convert to HSV and measure fraction of pixels with saturation > threshold
    and reasonable brightness. Works with Pillow only (no NumPy needed).
    """
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            # Optional: crop tiny borders where upscalers sometimes leave artifacts
            if crop_border > 0:
                w, h = im.size
                if w > crop_border*2 and h > crop_border*2:
                    im = im.crop((crop_border, crop_border, w - crop_border, h - crop_border))
            hsv = im.convert("HSV")
            h, s, v = hsv.split()
            s_data = s.getdata()
            v_data = v.getdata()

            colored = 0
            total = 0
            # Ignore *very* dark pixels to reduce false positives
            for sv, vv in zip(s_data, v_data):
                total += 1
                if sv > sat_thresh and vv > 32:   # vv>32 avoids near-black noise
                    colored += 1
            if total == 0:
                return False
            fraction = colored / total
            return fraction > color_frac
    except Exception as e:
        # If detection fails, fail-safe to NOT blocking the image
        print(f"⚠️ Color detection failed for {path}: {e}")
        return False

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
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,   # keep JSON
                stderr=None,              # let human logs hit journal
                text=True,
                env=ENV,
            )
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

                # ⬇️ NEW: enforce grayscale only for coloring jobs (not for covers)
                mode = (result.get("mode") or "").lower()
                # Treat either the job flag or the generator's returned mode as "cover"
                is_cover = bool(job.get("cover_mode") or ((result.get("mode") or "").lower() == "cover"))

                if ENFORCE_GRAYSCALE and not is_cover and mode == "coloring":
                    attempts = 0
                    if final_path and os.path.isfile(final_path) and _is_color_image(final_path):
                        print(f"⚠️ Color detected in {final_path}. Beginning monochrome retries...")
                    while final_path and os.path.isfile(final_path) and _is_color_image(final_path):
                        attempts += 1
                        if attempts > MAX_COLOR_RETRIES:
                            raise ValueError(f"Color detected after {MAX_COLOR_RETRIES} retries")

                        # Strengthen the prompt to force B/W line art
                        base_prompt = job["prompt"]
                        enforced_prompt = (
                            f"{base_prompt}, black and white, pure line art, monochrome, high-contrast, "
                            f"no color, no tint, white background"
                        )

                        # Rebuild the command with the same OUTPUT and stronger prompt
                        cmd_retry = [
                            PYTHON_BIN,
                            RUN_FLUX_SCRIPT,
                            "--prompt", enforced_prompt,
                            "--output", f"{job_id}.png",
                            "--output_dir", OUTPUT_DIR,
                            "--steps", str(job["steps"]),
                            "--guidance_scale", str(job["guidance_scale"])
                        ]
                        if job.get("autotune"):
                            cmd_retry.append("--autotune")
                        if job.get("adults"):
                            cmd_retry.append("--adults")
                        if job.get("cover_mode"):
                            cmd_retry.append("--cover_mode")

                        print(f"🔁 Retry {attempts}/{MAX_COLOR_RETRIES} with enforced monochrome...")

                        # Run generator again (same filename)
                        proc2 = subprocess.Popen(
                            cmd_retry,
                            stdout=subprocess.PIPE,   # keep JSON
                            stderr=None,              # let human logs hit journal
                            text=True,
                            env=ENV,
                        )
                        stdout2, stderr2 = proc2.communicate()

                        if proc2.returncode != 0:
                            print(f"❌ Retry failed (code {proc2.returncode}). STDERR:\n{stderr2}")
                            raise ValueError("Monochrome retry failed")

                        # Parse new JSON
                        json_match2 = re.search(r"\{.*\}", stdout2, re.DOTALL)
                        if not json_match2:
                            raise ValueError("Retry did not return JSON")

                        result2 = json.loads(json_match2.group(0))
                        if result2.get("error"):
                            raise ValueError(f"Retry error: {result2['error']}")

                        # Update final_path/width/height/upscaled to the retried output
                        final_path = result2.get("file") or final_path
                        upscaled   = result2.get("upscaled", upscaled)
                        width      = result2.get("width") or width
                        height     = result2.get("height") or height

                        print(f"✅ Retry produced {final_path} ({width}x{height})")
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
