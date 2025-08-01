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

# KDP Constants
BLEED_INCH = 0.125
DPI = 300
INTERIOR_WIDTH = 2550    # 8.5" * 300dpi
INTERIOR_HEIGHT = 3300   # 11" * 300dpi

# RealSR Model Path
REALSR_MODEL_PATH = "/usr/local/bin/models/models-DF2K"


def calculate_cover_dimensions(page_count, trim_width=8.5, trim_height=11):
    """Calculate KDP cover wrap dimensions in pixels."""
    spine_in = round(page_count / 444, 3)  # spine thickness in inches
    width_in = (trim_width * 2) + spine_in + (BLEED_INCH * 2)
    height_in = trim_height + (BLEED_INCH * 2)
    return int(width_in * DPI), int(height_in * DPI)


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
            steps.append(4)  # Force 4x because DF2K supports only 4x

        print(f"✅ RealSR Upscale Plan: Steps={steps}, Target Scale≈{final_width / current_w:.2f}")

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

        # ✅ After RealSR, force resize to exact final dimensions
        img = Image.open(output_path)
        img = img.resize((final_width, final_height), Image.LANCZOS)
        img.save(output_path)
        print(f"✅ Final image resized to exact: {final_width}x{final_height}")

        return True
    except Exception as e:
        print(f"⚠️ Multi-step upscale failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="FBN Publishing Image Generator (Flux Schnell + RealSR)")

    # Core options
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative_prompt", type=str,
                        default="blur, background clutter, text, watermark, trademarked, copyrighted")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--guidance_scale", type=float, default=4.5)

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

    # Upscale toggle
    parser.add_argument("--no-upscale", action="store_true")

    args = parser.parse_args()

    # ✅ Thread tuning
    if args.autotune:
        logical_cores = multiprocessing.cpu_count()
        tuned_threads = max(4, int(logical_cores * 0.75))
        torch.set_num_threads(tuned_threads)
        print(f"🧠 Auto-tuned threads: {tuned_threads}/{logical_cores}")
    else:
        torch.set_num_threads(args.threads)

    # ✅ Set final and starting dimensions automatically
    if args.cover_mode:
        if not args.page_count:
            raise ValueError("--page_count is required for cover_mode")
        final_width, final_height = calculate_cover_dimensions(args.page_count)
        start_width = 1024
        start_height = int(start_width * (final_height / final_width))
    else:
        final_width, final_height = INTERIOR_WIDTH, INTERIOR_HEIGHT
        start_width, start_height = 848, 1088

    print(f"📏 Final Size: {final_width}x{final_height}, Starting Size: {start_width}x{start_height}")

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
    pipe = DiffusionPipeline.from_pretrained(args.model_path, torch_dtype=torch.float32, use_safetensors=True)
    pipe.to("cpu")
    pipe.enable_attention_slicing()

    print(f"⏳ Generating image for prompt: {full_prompt}")

    start = time.time()
    image = pipe(
        prompt=full_prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        height=start_height,
        width=start_width
    ).images[0]
    image.save(output_path)
    end = time.time()

    # ✅ Upscale dynamically for covers
    final_output_path = output_path
    upscaled_done = False

    if not args.no_upscale and final_width and final_height:
        print(f"🔍 Upscaling to {final_width}×{final_height} using RealSR (multi-step)...")
        if upscale_image_multistep(output_path, upscaled_path, final_width, final_height):
            try:
                os.remove(output_path)
                shutil.move(upscaled_path, output_path)
                upscaled_done = True
                final_output_path = output_path
            except Exception as e:
                print(f"⚠️ Failed to replace original after upscale: {e}")

    # ✅ Build JSON result
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
        "mode": "cover" if args.cover_mode else "coloring",
        "time_sec": round(end - start, 2)
    }

    print(json.dumps(result))


if __name__ == "__main__":
    main()
