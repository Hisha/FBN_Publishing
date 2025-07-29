import argparse
import os
import torch
import time
import json
import subprocess
from datetime import datetime
from diffusers import DiffusionPipeline
import multiprocessing
import re

# Target print size for KDP interior pages
KDP_WIDTH = 2550   # 8.5 inches * 300 DPI
KDP_HEIGHT = 3300  # 11 inches * 300 DPI

def upscale_image(input_path, output_path):
    """
    Upscales an image to KDP specs (2550x3300) using Real-ESRGAN.
    Assumes realesrgan-ncnn-vulkan binary is installed and available in PATH.
    """
    try:
        cmd = [
            "realesrgan-ncnn-vulkan",  # Adjust if using CPU binary (realesrgan-ncnn-vulkan or realesrgan-ncnn)
            "-i", input_path,
            "-o", output_path,
            "-s", "4"  # 4x upscale
        ]
        subprocess.run(cmd, check=True)

        # Verify file exists after upscale
        if os.path.exists(output_path):
            return True
        else:
            return False
    except Exception as e:
        print(f"⚠️ Upscale failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run FLUX.1-schnell-Free image generation (FBN Publishing with KDP Upscale)")

    # Core generation options
    parser.add_argument("--prompt", type=str, required=True, help="Main subject or theme")
    parser.add_argument("--negative_prompt", type=str,
                        default="color, colored, blur, background clutter, shading, shadow, gradients, text, watermark",
                        help="Negative prompt to avoid unwanted features")
    parser.add_argument("--steps", type=int, default=4, help="Number of inference steps (default 4)")
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="Classifier-free guidance scale")

    # Image size and aspect
    parser.add_argument("--height", type=int, default=None, help="Image height in pixels")
    parser.add_argument("--width", type=int, default=None, help="Image width in pixels")
    parser.add_argument("--preset", type=str, choices=["square", "portrait"], default="portrait",
                        help="Aspect ratio preset: 'square' (1024x1024) or 'portrait' (8.5x11)")

    # Output handling
    parser.add_argument("--output", type=str, default=None, help="Output image filename")
    parser.add_argument("--output_dir", type=str, default="~/FluxImages/", help="Directory to save outputs to")

    # Performance options
    parser.add_argument("--threads", type=int, default=8, help="Manual thread count (ignored if --autotune is used)")
    parser.add_argument("--autotune", action="store_true", help="Auto-select optimal thread count for CPU")

    # Advanced
    parser.add_argument("--model_path", type=str, default=os.path.expanduser("~/FBN_publishing/"),
                        help="Path to local model folder")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose logs (for automation)")
    parser.add_argument("--adults", action="store_true", help="If set, generates more intricate detail for adults")

    # Upscale control
    parser.add_argument("--no-upscale", action="store_true", help="Disable upscaling to KDP print size")

    args = parser.parse_args()

    # ==========================
    # ✅ Thread selection
    # ==========================
    if args.autotune:
        logical_cores = multiprocessing.cpu_count()
        tuned_threads = max(4, int(logical_cores * 0.75))
        torch.set_num_threads(tuned_threads)
        if not args.quiet:
            print(f"🧠 Auto-tuned threads: {tuned_threads} of {logical_cores}")
    else:
        torch.set_num_threads(args.threads)
        if not args.quiet:
            print(f"🧠 Using manual thread count: {args.threads}")

    # ==========================
    # ✅ Handle aspect ratio presets
    # ==========================
    if args.preset == "square":
        if args.height is None: args.height = 1024
        if args.width is None: args.width = 1024
    elif args.preset == "portrait":
        if args.height is None: args.height = 1088  # Rounded to multiple of 16
        if args.width is None: args.width = 848    # Rounded to multiple of 16

    # ==========================
    # ✅ Build full prompt
    # ==========================
    base_template = (
        "black and white line art, clean bold outlines, highly detailed, "
        "no shading, no gradients, no color, coloring book illustration, "
        "plain white background, high contrast, ink drawing style"
    )

    if args.adults:
        detail_tag = ", for adults, very intricate design"
    else:
        detail_tag = ", for kids, simple and fun"

    full_prompt = f"{args.prompt}, {base_template}{detail_tag}"

    # ==========================
    # ✅ Random seed
    # ==========================
    if args.seed is not None:
        torch.manual_seed(args.seed)

    # ==========================
    # ✅ Prepare output paths
    # ==========================
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prompt = re.sub(r'[^a-z0-9_]+', '_', args.prompt[:40].lower())
    base_filename = args.output if args.output else f"{timestamp}_{safe_prompt}.png"
    output_path = os.path.join(output_dir, base_filename)

    # Upscaled path
    upscaled_path = output_path.replace(".png", "_upscaled.png")

    # ==========================
    # ✅ Load model
    # ==========================
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

    # ==========================
    # ✅ Upscale if enabled
    # ==========================
    upscaled_done = False
    if not args.no_upscale:
        if not args.quiet:
            print(f"🔍 Upscaling to KDP size ({KDP_WIDTH}x{KDP_HEIGHT})...")
        upscaled_done = upscale_image(output_path, upscaled_path)

    # ==========================
    # ✅ JSON output for automation
    # ==========================
    result = {
        "status": "success",
        "file": upscaled_path if upscaled_done else output_path,
        "prompt": full_prompt,
        "negative_prompt": args.negative_prompt,
        "seed": args.seed,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "height": args.height,
        "width": args.width,
        "upscaled": upscaled_done,
        "time_sec": round(end - start, 2)
    }

    print(json.dumps(result))

if __name__ == "__main__":
    main()
