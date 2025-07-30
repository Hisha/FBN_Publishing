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

# Target print size for KDP interior pages
KDP_WIDTH = 2550   # 8.5 inches * 300 DPI
KDP_HEIGHT = 3300  # 11 inches * 300 DPI

# Absolute path to RealSR models folder
REALSR_MODEL_PATH = "/usr/local/bin/models/models-DF2K"

def upscale_image(input_path, output_path):
    """
    Upscales an image using RealSR NCNN Vulkan with absolute model path.
    Assumes realsr-ncnn-vulkan binary is installed and models are in REALSR_MODEL_PATH.
    """
    try:
        cmd = [
            "realsr-ncnn-vulkan",
            "-i", input_path,
            "-o", output_path,
            "-s", "4",                         # 4x upscale
            "-m", REALSR_MODEL_PATH,          # Absolute model path
            "-g", "-1"                        # Force CPU
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
    parser.add_argument("--prompt", type=str, required=True, help="Main subject or theme")
    parser.add_argument("--negative_prompt", type=str,
                        default="blur, background clutter, text, watermark, trademarked, copyrighted",
                        help="Negative prompt to avoid unwanted features")
    parser.add_argument("--steps", type=int, default=4, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="Classifier-free guidance scale")

    # Image size
    parser.add_argument("--height", type=int, default=None, help="Image height in px")
    parser.add_argument("--width", type=int, default=None, help="Image width in px")
    parser.add_argument("--preset", type=str, choices=["square", "portrait"], default="portrait",
                        help="Aspect ratio preset: 'square' or 'portrait'")

    # Output
    parser.add_argument("--output", type=str, default=None, help="Output filename")
    parser.add_argument("--output_dir", type=str, default="~/FluxImages/", help="Directory for saving")

    # Performance
    parser.add_argument("--threads", type=int, default=8, help="Manual thread count")
    parser.add_argument("--autotune", action="store_true", help="Auto-select optimal thread count")

    # Advanced flags
    parser.add_argument("--model_path", type=str, default=os.path.expanduser("~/FBN_publishing/"),
                        help="Path to local model folder")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose logs")
    parser.add_argument("--adults", action="store_true", help="Intricate design for adults")
    parser.add_argument("--cover_mode", action="store_true", help="Enable full-color cover art mode")

    # Upscale toggle
    parser.add_argument("--no-upscale", action="store_true", help="Disable KDP upscaling")

    args = parser.parse_args()

    # ✅ Threads
    if args.autotune:
        logical_cores = multiprocessing.cpu_count()
        tuned_threads = max(4, int(logical_cores * 0.75))
        torch.set_num_threads(tuned_threads)
        if not args.quiet:
            print(f"🧠 Auto-tuned threads: {tuned_threads}/{logical_cores}")
    else:
        torch.set_num_threads(args.threads)
        if not args.quiet:
            print(f"🧠 Using manual thread count: {args.threads}")

    # ✅ Aspect ratio
    if args.preset == "square":
        args.height = args.height or 1024
        args.width = args.width or 1024
    elif args.preset == "portrait":
        args.height = args.height or 1088
        args.width = args.width or 848

    # ✅ Build prompt logic
    if args.cover_mode:
        full_prompt = args.prompt  # Full color cover, no restrictions
    else:
        base_template = (
            "black and white line art, clean bold outlines, highly detailed, "
            "no shading, no gradients, no color, coloring book illustration, "
            "plain white background, high contrast, ink drawing style"
        )
        detail_tag = ", for adults, very intricate design" if args.adults else ", for kids, simple and fun"
        full_prompt = f"{args.prompt}, {base_template}{detail_tag}"

        # Add strong negatives to prevent accidental color in coloring pages
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

    # ✅ Upscale if enabled
    final_output_path = output_path
    upscaled_done = False

    if not args.no_upscale:
        if not args.quiet:
            print(f"🔍 Upscaling to KDP size using RealSR...")
        if upscale_image(output_path, upscaled_path):
            try:
                os.remove(output_path)  # delete original
                shutil.move(upscaled_path, output_path)  # rename upscaled
                upscaled_done = True
                final_output_path = output_path
            except Exception as e:
                print(f"⚠️ Failed to replace original with upscaled: {e}")
        else:
            if not args.quiet:
                print("⚠️ Upscale failed, keeping original image.")

    # ✅ Build JSON result
    result = {
        "status": "success",
        "file": final_output_path,
        "prompt": full_prompt,
        "negative_prompt": args.negative_prompt,
        "seed": args.seed,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "height": args.height,
        "width": args.width,
        "upscaled": upscaled_done,
        "mode": "cover" if args.cover_mode else "coloring",
        "time_sec": round(end - start, 2)
    }

    print(json.dumps(result))

if __name__ == "__main__":
    main()
