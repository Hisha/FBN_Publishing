import argparse
import os
import torch
import time
import json
import subprocess
import shutil
import sys
from datetime import datetime
from diffusers import DiffusionPipeline
import multiprocessing
import re
from PIL import Image, ImageDraw, ImageFilter, ImageOps
import numpy as np

# =========================
# KDP & Layout Constants
# =========================
BLEED_INCH = 0.125
DPI = 300
INTERIOR_WIDTH = 2550    # 8.5" * 300dpi
INTERIOR_HEIGHT = 3300   # 11" * 300dpi

# RealSR Model Path
REALSR_MODEL_PATH = "/usr/local/bin/models/models-DF2K"

# =========================
# Color Guard (pre-processing)
# =========================
COLOR_CROP_BORDER_PX = int(os.getenv("COLOR_CROP_BORDER_PX", "8"))
COLOR_Y_MIN = int(os.getenv("COLOR_Y_MIN", "32"))
COLOR_Y_MAX = int(os.getenv("COLOR_Y_MAX", "240"))
COLOR_DELTA_THRESH = int(os.getenv("COLOR_DELTA_THRESH", "36"))
COLOR_FRACTION = float(os.getenv("COLOR_FRACTION", "0.01"))
DEBUG_COLOR_GUARD = os.getenv("DEBUG_COLOR_GUARD", "0") == "1"

BIN_THRESH = int(os.getenv("BIN_THRESH", "200"))    # 0..255; higher => fewer grays become black
THICKEN_SIZE = int(os.getenv("THICKEN_SIZE", "3"))  # 1/3/5... MaxFilter kernel size

SCRUB_WATERMARK = os.getenv("SCRUB_WATERMARK", "1") == "1"
WATERMARK_W = int(os.getenv("WATERMARK_W", "420"))   # ~1.4" at 300dpi
WATERMARK_H = int(os.getenv("WATERMARK_H", "160"))   # ~0.53"

# =========================
# Utilities
# =========================
def _is_color_pil(img: Image.Image) -> bool:
    """Return True if the image has meaningful color; False if it's monochrome enough."""
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg
    else:
        img = img.convert("RGB")

    w, h = img.size
    c = COLOR_CROP_BORDER_PX
    if c and w > 2*c and h > 2*c:
        img = img.crop((c, c, w - c, h - c))

    arr = np.asarray(img, dtype=np.int16)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    # Luma (BT.601)
    y = (299*r + 587*g + 114*b) // 1000
    mid = (y > COLOR_Y_MIN) & (y < COLOR_Y_MAX)
    total = int(mid.sum())
    if total == 0:
        if DEBUG_COLOR_GUARD:
            print("🔧 ColorGuard: no midtone pixels → MONO", file=sys.stderr, flush=True)
        return False

    delta = np.maximum.reduce([np.abs(r-g), np.abs(g-b), np.abs(r-b)])
    colored = int(((delta > COLOR_DELTA_THRESH) & mid).sum())
    frac = colored / total

    if DEBUG_COLOR_GUARD:
        print(
            f"🔧 ColorGuard: delta>{COLOR_DELTA_THRESH}, frac>{COLOR_FRACTION} → colored={colored}/{total} ({frac:.5f})",
            file=sys.stderr, flush=True
        )
    return frac > COLOR_FRACTION


def _scrub_watermark_corners(path: str):
    if not SCRUB_WATERMARK:
        return
    try:
        im = Image.open(path).convert("RGB")
        w, h = im.size
        d = ImageDraw.Draw(im)
        # bottom-right and bottom-left (most common)
        d.rectangle([w - WATERMARK_W, h - WATERMARK_H, w, h], fill=(255, 255, 255))
        d.rectangle([0, h - WATERMARK_H, WATERMARK_W, h], fill=(255, 255, 255))
        im.save(path)
        print("🧽 Watermark corners scrubbed", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"⚠️ Watermark scrub failed: {e}", file=sys.stderr, flush=True)


def binarize_and_thicken(pil_img: Image.Image) -> Image.Image:
    """Convert to clean black/white line art and thicken lines."""
    g = ImageOps.autocontrast(pil_img.convert("L"))
    bw = g.point(lambda v: 0 if v < BIN_THRESH else 255, mode="1").convert("L")
    if THICKEN_SIZE >= 3 and THICKEN_SIZE % 2 == 1:
        bw = bw.filter(ImageFilter.MaxFilter(THICKEN_SIZE))
    bw = bw.filter(ImageFilter.MinFilter(3))
    return bw.convert("RGB")


def resize_to_exact_lineart(img: Image.Image, w: int, h: int) -> Image.Image:
    r = img.resize((w, h), Image.LANCZOS)
    return binarize_and_thicken(r)

# =========================
# Cover dimension calc
# =========================
def calculate_cover_dimensions(page_count, trim_width=8.5, trim_height=11):
    """Calculate KDP cover wrap dimensions in pixels."""
    spine_in = round(page_count / 444, 3)  # KDP spine thickness approximation
    width_in = (trim_width * 2) + spine_in + (BLEED_INCH * 2)
    height_in = trim_height + (BLEED_INCH * 2)
    return int(width_in * DPI), int(height_in * DPI)

# =========================
# RealSR upscale (covers only)
# =========================
def upscale_image_multistep(input_path, output_path, final_width, final_height):
    try:
        img = Image.open(input_path)
        current_w, current_h = img.size
        scale_factor = final_width / current_w

        steps = []
        while scale_factor > 4:
            steps.append(4)
            scale_factor /= 4
        if scale_factor > 1:
            steps.append(4)  # DF2K supports only 4x

        print(
            f"✅ RealSR Upscale Plan: Steps={steps}, Target Scale≈{final_width / current_w:.2f}",
            file=sys.stderr, flush=True
        )

        temp_input = input_path
        for i, s in enumerate(steps):
            temp_output = output_path if i == len(steps) - 1 else temp_input.replace(".png", f"_x{s}_{i}.png")
            cmd = [
                "realsr-ncnn-vulkan",
                "-i", temp_input,
                "-o", temp_output,
                "-s", str(s),
                "-m", REALSR_MODEL_PATH,
                "-g", "-1"
            ]
            subprocess.run(cmd, check=True)
            if temp_input != input_path and os.path.exists(temp_input):
                os.remove(temp_input)
            temp_input = temp_output

        # Force resize to exact final dimensions
        img = Image.open(output_path)
        img = img.resize((final_width, final_height), Image.LANCZOS)
        img.save(output_path)
        print(f"✅ Final image resized to exact: {final_width}x{final_height}", file=sys.stderr, flush=True)

        return True
    except Exception as e:
        print(f"⚠️ Multi-step upscale failed: {e}", file=sys.stderr, flush=True)
        return False

# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(description="FBN Publishing Image Generator (Flux Schnell; RealSR for covers only)")

    # Core options
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative_prompt", type=str,
                        default="blur, background clutter, text, watermark, trademarked, copyrighted")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--guidance_scale", type=float, default=3.5)

    # Cover mode
    parser.add_argument("--page_count", type=int, default=None, help="Total pages for cover size calculation")

    # Output
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="~/FluxImages/")

    # Performance
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--autotune", action="store_true")

    # Advanced flags
    parser.add_argument("--model_path", type=str, default=os.path.expanduser("~/FBN_publishing/"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--adults", action="store_true")
    parser.add_argument("--cover_mode", action="store_true")

    args = parser.parse_args()

    # Thread tuning
    if args.autotune:
        logical_cores = multiprocessing.cpu_count()
        tuned_threads = max(4, int(logical_cores * 0.75))
        torch.set_num_threads(tuned_threads)
        print(f"🧠 Auto-tuned threads: {tuned_threads}/{logical_cores}", file=sys.stderr, flush=True)
    else:
        torch.set_num_threads(args.threads)

    # Final / starting dimensions
    if args.cover_mode:
        if not args.page_count:
            raise ValueError("--page_count is required for cover_mode")
        final_width, final_height = calculate_cover_dimensions(args.page_count)
        start_width = 1024
        start_height = int(start_width * (final_height / final_width))
    else:
        final_width, final_height = INTERIOR_WIDTH, INTERIOR_HEIGHT
        start_width, start_height = 848, 1088

    print(f"📏 Final Size: {final_width}x{final_height}, Starting Size: {start_width}x{start_height}", file=sys.stderr, flush=True)

    # =========================
    # Build prompt
    # =========================
    # Neutral base template (works for both kids and adults)
    base_template_kids = (
        "black and white line art, vector-like, bold thick outlines (4-6 px at 300 dpi), "
        "minimal interior lines, large simple shapes, flat white background, single scene, centered composition, "
        "no text or numbers, simple and fun, ages 4-8"
    )
    base_template_adult = (
        "black and white line art, vector-like, bold outlines, high line density, complex symmetrical pattern, "
        "intricate ornamental motifs, flat white background, single scene, centered composition, no text or numbers"
    )

    if args.cover_mode:
        full_prompt = (
            f"{args.prompt}, full wraparound book cover layout, "
            "leave room for title text on front, spine, and description space on back, "
            "hand-colored crayon and colored pencil style, visible wax texture, uneven coloring, "
            "soft artistic feel, vibrant but not overly saturated colors"
        )
        args.negative_prompt += (
            ", photorealistic, realistic, hyper-realistic, CGI, logo, signature, caption, url, "
            "website, letters, typography"
        )
    else:
        style = base_template_adult if args.adults else base_template_kids
        full_prompt = f"{args.prompt}, {style}"
        args.negative_prompt += (
            ", color, colored, gradients, grayscale tones, grey wash, texture, noise, "
            "crosshatching, hatching, stippling, halftone, shading, drop shadow, glow, "
            "3d lighting, airbrush, sketchy lines, pencil shading, graphite, charcoal, "
            "background clutter, logo, watermark, signature, caption, url, website, letters, typography"
        )

    # Seed
    if args.seed is not None:
        torch.manual_seed(args.seed)

    # Output paths
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prompt = re.sub(r'[^a-z0-9_]+', '_', args.prompt[:40].lower())
    base_filename = args.output if args.output else f"{timestamp}_{safe_prompt}.png"
    output_path = os.path.join(output_dir, base_filename)
    upscaled_path = output_path.replace(".png", "_upscaled.png")

    # Load model
    pipe = DiffusionPipeline.from_pretrained(args.model_path, torch_dtype=torch.float32, use_safetensors=True)
    pipe.to("cpu")
    pipe.enable_attention_slicing()

    print(f"⏳ Generating image for prompt: {full_prompt}", file=sys.stderr, flush=True)

    start = time.time()
    image = pipe(
        prompt=full_prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        height=start_height,
        width=start_width
    ).images[0]

    # COLOR GUARD (interiors only). Run BEFORE any upscale.
    if not args.cover_mode:
        if _is_color_pil(image):
            print("⚠️ Base render shows color tint; forcing grayscale", file=sys.stderr, flush=True)
            image = image.convert("L").convert("RGB")
        else:
            print("✅ Base render passes color guard", file=sys.stderr, flush=True)

    # Save the base image
    image.save(output_path)
    end = time.time()

    # =========================
    # Upscale / finalize
    # =========================
    final_output_path = output_path
    upscaled_done = False

    if args.cover_mode:
        # Covers: Use RealSR, then exact resize
        if final_width and final_height:
            print(f"🔍 Upscaling cover to {final_width}×{final_height} using RealSR (multi-step)...", file=sys.stderr, flush=True)
            if upscale_image_multistep(output_path, upscaled_path, final_width, final_height):
                try:
                    if os.path.exists(output_path):
                        os.remove(output_path)
                    shutil.move(upscaled_path, output_path)
                    upscaled_done = True
                    final_output_path = output_path
                except Exception as e:
                    print(f"⚠️ Cover: failed to replace original after upscale: {e}", file=sys.stderr, flush=True)
            else:
                # Fallback to Pillow resize
                im_final = image.resize((final_width, final_height), Image.LANCZOS)
                im_final.save(final_output_path)
                print("ℹ️ RealSR failed; used Pillow resize for cover", file=sys.stderr, flush=True)
    else:
        # Interiors: Exact resize + binarize (best for crisp line art)
        try:
            im_final = resize_to_exact_lineart(image, final_width, final_height)
            im_final.save(final_output_path)
            print("✅ Interior exact resize + binarize complete", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"⚠️ Interior resize+binarize failed: {e}", file=sys.stderr, flush=True)
        _scrub_watermark_corners(final_output_path)

    # =========================
    # JSON result
    # =========================
    result = {
        "status": "success",
        "file": final_output_path,
        "prompt": full_prompt,
        "negative_prompt": args.negative_prompt,
        "seed": args.seed,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "height": final_height,
        "width": final_width,
        "upscaled": upscaled_done,
        "mode": "cover" if args.cover_mode else ("adult" if args.adults else "kids"),
        "time_sec": round(end - start, 2)
    }

    print(json.dumps(result))


if __name__ == "__main__":
    main()
