import os
import uuid
import shutil
import threading
import subprocess
from datetime import datetime
from fastapi import FastAPI, Request, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from dotenv import load_dotenv
from auth import login_required, router as auth_router
from db import init_db, add_job, get_jobs, update_job_status, get_job_by_id

# Load env vars
load_dotenv()

# Config
SECRET_KEY = os.getenv("SESSION_SECRET", "supersecret")
BASE_OUTPUT_DIR = os.path.expanduser("~/FluxImages")
MODEL_PATH = os.path.expanduser("~/FBN_publishing/")
API_TOKEN = os.getenv("API_TOKEN", "changeme")

# FastAPI setup
app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(auth_router)
templates = Jinja2Templates(directory="templates")

# Init DB
init_db()

# Job Queue
job_lock = threading.Lock()

def process_jobs():
    while True:
        jobs = get_jobs(status="queued")
        if jobs:
            job = jobs[0]
            update_job_status(job["id"], "running")
            try:
                run_job(job)
                update_job_status(job["id"], "done", job["file_path"])
            except Exception as e:
                print(f"Job failed: {e}")
                update_job_status(job["id"], "error")
        else:
            # No jobs, sleep briefly
            import time
            time.sleep(5)

def run_job(job):
    """Run run_flux.py with provided job details."""
    cmd = [
        "python", "run_flux.py",
        "--prompt", job["prompt"],
        "--autotune",
        "--preset", "portrait"
    ]
    if job["adults"]:
        cmd.append("--adults")
    if not job["upscale"]:
        cmd.append("--no-upscale")
    if job["filename"]:
        cmd.extend(["--output", job["filename"]])
    if job["output_dir"]:
        cmd.extend(["--output_dir", BASE_OUTPUT_DIR])  # Save to default, then copy later

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)

    # Parse JSON from output
    import json
    data = json.loads(result.stdout.strip().split("\n")[-1])
    generated_file = data["file"]

    final_path = generated_file
    if job["output_dir"] and job["filename"]:
        os.makedirs(job["output_dir"], exist_ok=True)
        dest_path = os.path.join(job["output_dir"], job["filename"])
        shutil.copy(generated_file, dest_path)
        final_path = dest_path

    job["file_path"] = final_path

# Start background worker
threading.Thread(target=process_jobs, daemon=True).start()

# Routes
@app.get("/", response_class=HTMLResponse)
@login_required
async def index(request: Request):
    jobs = get_jobs(limit=10)
    return templates.TemplateResponse("index.html", {"request": request, "jobs": jobs})

@app.get("/jobs", response_class=HTMLResponse)
@login_required
async def jobs_page(request: Request):
    jobs = get_jobs(limit=50)
    return templates.TemplateResponse("jobs.html", {"request": request, "jobs": jobs})

@app.post("/submit")
@login_required
async def submit_job(request: Request,
                     prompt: str = Form(...),
                     adults: bool = Form(False),
                     upscale: bool = Form(False),
                     output_dir: str = Form(None),
                     filename: str = Form(None)):
    job_id = str(uuid.uuid4())
    add_job(job_id, prompt, adults, upscale, output_dir, filename)
    return RedirectResponse("/", status_code=303)

@app.post("/generate/json")
async def generate_json(request: Request):
    auth = request.headers.get("Authorization", "")
    if auth != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    body = await request.json()
    prompt = body.get("prompt")
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    adults = body.get("adults", False)
    upscale = body.get("upscale", True)
    output_dir = body.get("output_dir")
    filename = body.get("filename")

    job_id = str(uuid.uuid4())
    add_job(job_id, prompt, adults, upscale, output_dir, filename)
    return JSONResponse({"status": "queued", "job_id": job_id, "message": "Job added to queue"})
