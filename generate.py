# generate.py
import argparse
import logging
from datetime import datetime
from PIL import Image
import os
from huggingface_hub import snapshot_download

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS
from wan.utils.utils import save_video

# Ensure HuggingFace token is set
if "HF_TOKEN" not in os.environ:
    raise ValueError("HF_TOKEN not found. Please set it in RunPod environment variables.")

# Ensure temp dir exists
tmp_dir = os.environ.get("TMPDIR", "/workspace/tmp")
os.makedirs(tmp_dir, exist_ok=True)


def _parse_args():
    parser = argparse.ArgumentParser(description="Wan2.2 I2V-A8B (Image + Prompt → Video)")

    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for generation")
    parser.add_argument("--image", type=str, required=True, help="Path to input reference image")
    parser.add_argument("--size", type=str, default="1280*720", choices=list(SIZE_CONFIGS.keys()))

    parser.add_argument("--ckpt_dir", type=str, default="Wan-AI/Wan2.2-I2V-A8B")

    # Video settings
    parser.add_argument("--frame_num", type=int, default=81, help="Frames per video")
    parser.add_argument("--fps", type=int, default=16, help="Frames per second")
    parser.add_argument("--sample_steps", type=int, default=25)
    parser.add_argument("--sample_shift", type=float, default=None)
    parser.add_argument("--sample_guide_scale", type=float, default=None)
    parser.add_argument("--base_seed", type=int, default=42)

    # Memory
    parser.add_argument("--offload_model", action="store_true", help="Enable model offloading (saves VRAM)")

    parser.add_argument("--save_file", type=str, default=None)
    return parser.parse_args()


def generate(args):
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    cfg = WAN_CONFIGS["i2v-A14B"]  # A8B uses same config keys

    # Download model weights if not cached
    if "/" in args.ckpt_dir:
        hf_token = os.getenv("HF_TOKEN")
        logging.info(f"Downloading model from Hugging Face: {args.ckpt_dir}")
        local_dir = snapshot_download(
            repo_id=args.ckpt_dir,
            token=hf_token,
            cache_dir="/workspace/hf_models"  # within your 20GB container disk
        )
        args.ckpt_dir = local_dir

    logging.info(f"Loading Wan2.2 I2V-A8B from {args.ckpt_dir}")

    pipeline = wan.WanI2V(config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0)

    # Load input image
    image = Image.open(args.image).convert("RGB")

    # Generate video
    video = pipeline.generate(
        args.prompt,
        image,
        max_area=MAX_AREA_CONFIGS[args.size],
        frame_num=args.frame_num,
        shift=(args.sample_shift or cfg.sample_shift),
        sampling_steps=args.sample_steps,
        guide_scale=(args.sample_guide_scale or cfg.sample_guide_scale),
        seed=args.base_seed,
        offload_model=args.offload_model,
    )

    # Save result
    if args.save_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_file = f"i2v_A8B_{timestamp}.mp4"

    logging.info(f"Saving video to {args.save_file}")
    save_video(
        tensor=video[None],
        save_file=args.save_file,
        fps=args.fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )

    logging.info("✅ Finished generation")
    return args.save_file


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
