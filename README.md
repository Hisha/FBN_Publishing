# 🖼️ FBN Publishing Image Generation API

This project is a **FastAPI-based system** for managing **Flux Schnell CPU-powered image generation** with **RealSR upscaling** for high-resolution outputs. It supports **job queuing, automatic upscaling to KDP print size, and an admin dashboard** for managing jobs.

---

## ✅ Features

* **Flux Schnell CPU Integration**

  * Generates coloring book and cover images from text prompts using `diffusers`.
* **RealSR Upscaling**

  * Upscales images to **8.5x11 inches @ 300 DPI** (KDP print-ready size).
* **FastAPI Dashboard**

  * Job submission UI with **HTMX auto-refresh**.
  * Filter jobs by status (`queued`, `in_progress`, `done`, `failed`).
  * Search jobs by prompt keywords.
* **SQLite Database**

  * Persistent tracking of jobs, prompts, parameters, and results.
* **Custom Output Support**

  * Optional **custom filenames** and **output directories**.
* **Dynamic Resolution Updates**

  * Uses `ffprobe` (or fallback) to update final resolution after upscaling.
* **Authentication**

  * Admin login via session middleware.
* **JSON API**

  * Programmatic job submission for integration with **n8n** or other workflows.

---

## 📂 Project Structure

```
FBN_Publishing/
├── fbnp_api.py         # Main FastAPI application
├── job_queue.py        # Worker queue for processing jobs
├── run_flux.py         # Handles image generation and upscaling
├── templates/          # Jinja2 templates for dashboard
├── static/             # CSS/JS assets
├── db.py               # SQLite job management
└── README.md           # This file
```

---

## ⚙️ Requirements

* **Python 3.8+**
* **ffmpeg** (for resolution detection via `ffprobe`)
* **RealSR NCNN Vulkan** for image upscaling
  Download from: [Real-ESRGAN Releases](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/releases)
* CPU environment (no GPU required)

---

## 🔧 Installation

### 1. Clone the repository:

```bash
git clone https://github.com/yourusername/FBN_Publishing.git
cd FBN_Publishing
```

### 2. Create and activate virtual environment:

```bash
python3 -m venv FBNP_env
source FBNP_env/bin/activate
```

### 3. Upgrade pip:

```bash
pip install --upgrade pip
```

### 4. Install dependencies:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install diffusers transformers accelerate safetensors scipy sentencepiece protobuf
pip install pytz fastapi uvicorn python-multipart bcrypt itsdangerous python-dotenv
```

### 5. Install RealSR NCNN Vulkan:

* Download and extract the precompiled binary from: [Real-ESRGAN Releases](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/releases)
* Ensure `realsr-ncnn-vulkan` is in your system PATH.

---

## ▶️ Running the API

Start the FastAPI server:

```bash
uvicorn fbnp_api:app --host 0.0.0.0 --port 8000 --reload
```

Start the job worker process (in another terminal):

```bash
python job_queue.py
```

Access the dashboard in your browser:

```
http://<server-ip>:8000/fbnp
```

---

## 🛠️ Configuration

### Environment Variables:

Create a `.env` file in the project root:

```
SECRET_KEY=your_secret_key_here
N8N_API_TOKEN=your_token_here
```

### Output Directory:

Default: `~/FluxImages`
Can be overridden via form or JSON input.

---

## 🗓 API Endpoints

### Job Submission (HTML Form)

`POST /generate`
Submit a prompt from the dashboard UI.

### Job Submission (JSON API)

`POST /generate/json`
Example:

```json
{
  "prompt": "mystical woodland scene with fairies",
  "steps": 4,
  "guidance_scale": 3.5,
  "height": 1088,
  "width": 848,
  "adults": false,
  "cover_mode": true,
  "seed": 12345
}
```

### Get Jobs

`GET /jobs/json?status=done`
Returns recent jobs in JSON format.

---

## 🔖 Usage Notes

* **Coloring Pages vs Cover Mode**

  * Cover Mode → Enables full-color artwork.
  * Default → Black-and-white line art for coloring books.

* **Upscaling Behavior**

  * If upscaling succeeds, only the upscaled image is kept.
  * If upscaling fails, original image is retained.

* **Resolution Updates**

  * Final width/height detected via `ffprobe` and updated in the database.

---

## ✅ Example Prompts

* majestic medieval castle with high spires, surrounded by forests
* detailed steampunk airship with gears and propellers (with seed)
* vibrant coral reef palace with seashell arches (cover mode)
* whimsical woodland creatures in a magical forest (adult coloring page)
* epic desert pyramid with intricate carvings under twilight skies

---

## 🖼️ Screenshot

*(Add a screenshot of the dashboard once available)*

---

## 📌 To-Do / Future Enhancements

*

---

## ⚖️ License

MIT License – Use at your own risk.
