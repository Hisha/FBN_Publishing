import argparse
import os
import torch
import time
import json
import subprocess
import shutil
from datetime import datetime
from diffusers import DiffusionPipeline
import multiprocessing
import re
from PIL import Image

# KDP Specs
BLEED_INCH = 0.125
DPI = 300

# RealSR Models
REALSR_MODEL_PATH = "/usr/local/bin/models/models-DF2K"


def calculate_cover_dimensions(page_count, trim_width=8.5, trim_height=11):
    """Calculate KDP cover wrap dimensions in pixels."""
    spine_in = round(page_count / 444, 3)  # spine thickness in inches
    width_in = (trim_width * 2) + spine_in + (BLEED_INCH * 2)
    height_in = trim_height + (BLEED_INCH * 2)
    return int(width_in * DPI), int(height_in * DPI)  # width_px, height_px


def upscale_image(input_path, output_path, scale):
    """Upscales an image using RealSR NCNN Vulkan with dynamic scale."""
    try:
        cmd = [
            "realsr-ncnn-vulkan",
            "-i", input_path,
            "-o", output_path,
            "-s", str(scale),                    # dynamic scale
            "-m", REALSR_MODEL_PATH,
            "-g", "-1"                           # CPU mode
        ]
        subprocess.run(cmd, check=True)
        return os.path.exists(output_path)
    except FileNotFoundError:
        print("⚠️ realsr-ncnn-vulkan not found in PATH. Skipping upscale.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Upscale failed: {e}")
        return False
    except Exception as e:
        print(f"⚠️ Unexpected error during upscale: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="FBN Publishing image generator (Flux Schnell + RealSR)")

    # Core options
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative_prompt", type=str,
                        default="blur, background clutter, text, watermark, trademarked, copyrighted")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance_scale", type=float, default=3.5)

    # Image size
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--preset", type=str, choices=["square", "portrait"], default="portrait")

    # Cover-specific
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
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--adults", action="store_true")
    parser.add_argument("--cover_mode", action="store_true")

    # Upscale toggle
    parser.add_argument("--no-upscale", action="store_true")

    args = parser.parse_args()

    # ✅ Thread tuning
    if args.autotune:
        logical_cores = multiprocessing.cpu_count()
        tuned_threads = max(4, int(logical_cores * 0.75))
        torch.set_num_threads(tuned_threads)
        if not args.quiet:
            print(f"🧠 Auto-tuned threads: {tuned_threads}/{logical_cores}")
    else:
        torch.set_num_threads(args.threads)

    # ✅ Calculate cover ratio if page_count given
    final_width = None
    final_height = None
    if args.page_count and args.cover_mode:
        final_width, final_height = calculate_cover_dimensions(args.page_count)
        aspect_ratio = final_width / final_height
        # Start with smaller image at same ratio
        args.width = 1024
        args.height = int(args.width / aspect_ratio)
    else:
        # Fallback to normal coloring pages
        if args.preset == "square":
            args.height = args.height or 1024
            args.width = args.width or 1024
        else:
            args.height = args.height or 1088
            args.width = args.width or 848

    # ✅ Build prompt
    if args.cover_mode:
        full_prompt = (
            f"{args.prompt}, full wraparound book cover layout, "
            "leave room for title text on front, spine, and description space on back, "
            "hand-colored crayon and colored pencil style, visible wax texture, uneven coloring, "
            "soft artistic feel, vibrant but not overly saturated colors"
        )
        args.negative_prompt += ", photorealistic, realistic, hyper-realistic, CGI"
    else:
        base_template = (
            "black and white line art, clean bold outlines, highly detailed, "
            "no shading, no gradients, no color, coloring book illustration, "
            "plain white background, high contrast, ink drawing style"
        )
        detail_tag = ", for adults, very intricate design" if args.adults else ", for kids, simple and fun"
        full_prompt = f"{args.prompt}, {base_template}{detail_tag}"
        args.negative_prompt += ", color, colored, gradients"

    # ✅ Seed
    if args.seed is not None:
        torch.manual_seed(args.seed)

    # ✅ Output paths
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prompt = re.sub(r'[^a-z0-9_]+', '_', args.prompt[:40].lower())
    base_filename = args.output if args.output else f"{timestamp}_{safe_prompt}.png"
    output_path = os.path.join(output_dir, base_filename)
    upscaled_path = output_path.replace(".png", "_upscaled.png")

    # ✅ Load model
    pipe = DiffusionPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.float32,
        use_safetensors=True
    )
    pipe.to("cpu")
    pipe.enable_attention_slicing()

    if not args.quiet:
        print(f"⏳ Generating image for prompt: {full_prompt}")

    start = time.time()
    image = pipe(
        prompt=full_prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width
    ).images[0]
    image.save(output_path)
    end = time.time()

    # ✅ Upscale dynamically
    final_output_path = output_path
    upscaled_done = False
    if not args.no-upscale and final_width and final_height:
        if not args.quiet:
            print(f"🔍 Upscaling to {final_width}×{final_height} using RealSR...")
        try:
            img = Image.open(output_path)
            current_w, current_h = img.size
            scale = round(final_width / current_w, 2)
            if upscale_image(output_path, upscaled_path, scale):
                os.remove(output_path)
                shutil.move(upscaled_path, output_path)
                upscaled_done = True
        except Exception as e:
            print(f"⚠️ Failed dynamic upscale: {e}")

    # ✅ Build JSON result
    result = {
        "status": "success",
        "file": output_path,
        "prompt": full_prompt,
        "negative_prompt": args.negative_prompt,
        "seed": args.seed,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "height": final_height if final_height else args.height,
        "width": final_width if final_width else args.width,
        "upscaled": upscaled_done,
        "mode": "cover" if args.cover_mode else "coloring",
        "time_sec": round(end - start, 2)
    }

    print(json.dumps(result))


if __name__ == "__main__":
    main()
