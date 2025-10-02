# generate.py
import argparse
import logging
from datetime import datetime
from PIL import Image
import torch
import os
from huggingface_hub import snapshot_download

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS
from wan.utils.utils import save_video, merge_video_audio


EXAMPLE_PROMPT = {
    "t2v-A14B": {"prompt": "A cinematic shot of a dragon flying over snowy mountains."},
    "i2v-A14B": {"prompt": "A cat stands in a beach scene.", "image": "examples/cat.png"},
    "s2v-14B": {"prompt": "A singer is singing", "image": "examples/cat.png", "audio": "examples/talk.wav"}
}


def _parse_args():
    parser = argparse.ArgumentParser(description="Wan2.2 generation (T2V, I2V, S2V)")

    parser.add_argument("--task", type=str, default="t2v-A14B", choices=list(WAN_CONFIGS.keys()))
    parser.add_argument("--size", type=str, default="1280*720", choices=list(SIZE_CONFIGS.keys()))
    parser.add_argument("--ckpt_dir", type=str, required=True)

    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--audio", type=str, default=None)

    # Video settings
    parser.add_argument("--frame_num", type=int, default=81, help="Total frames (default ~5s at 16 FPS).")
    parser.add_argument("--fps", type=int, default=16, help="Frames per second.")
    parser.add_argument("--sample_steps", type=int, default=None)
    parser.add_argument("--sample_shift", type=float, default=None)
    parser.add_argument("--sample_guide_scale", type=float, default=None)

    parser.add_argument("--base_seed", type=int, default=-1)
    parser.add_argument("--offload_model", type=bool, default=True)
    parser.add_argument("--convert_model_dtype", action="store_true", default=False)

    parser.add_argument("--enable_tts", action="store_true", default=False)
    parser.add_argument("--tts_prompt_audio", type=str, default=None)
    parser.add_argument("--tts_prompt_text", type=str, default=None)
    parser.add_argument("--tts_text", type=str, default=None)

    parser.add_argument("--save_file", type=str, default=None)

    args = parser.parse_args()

    # Fill defaults if missing
    if args.prompt is None and args.task in EXAMPLE_PROMPT:
        args.prompt = EXAMPLE_PROMPT[args.task].get("prompt")
    if args.image is None and args.task in EXAMPLE_PROMPT:
        args.image = EXAMPLE_PROMPT[args.task].get("image")
    if args.audio is None and args.task == "s2v-14B":
        args.audio = EXAMPLE_PROMPT[args.task].get("audio")

    return args


def generate(args):
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    cfg = WAN_CONFIGS[args.task]

    # ✅ If ckpt_dir looks like a HF repo, download it
    if "/" in args.ckpt_dir:  # e.g. "Wan-AI/Wan2.2-T2V-A14B"
        logging.info(f"Downloading model from Hugging Face: {args.ckpt_dir}")
        local_dir = snapshot_download(
            repo_id=args.ckpt_dir,
            token=os.getenv("HF_TOKEN"),  # must be set in RunPod env
            cache_dir="/workspace/hf_models"
        )
        args.ckpt_dir = local_dir

    logging.info(f"Loading model for task {args.task} from {args.ckpt_dir}")

    # Select pipeline
    if args.task.startswith("t2v"):
        pipeline = wan.WanT2V(config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0)
        video = pipeline.generate(
            args.prompt,
            size=SIZE_CONFIGS[args.size],
            frame_num=args.frame_num,
            sampling_steps=(args.sample_steps or cfg.sample_steps),
            guide_scale=(args.sample_guide_scale or cfg.sample_guide_scale),
            seed=(args.base_seed if args.base_seed >= 0 else 42),
            offload_model=args.offload_model,
        )

    elif args.task.startswith("i2v"):
        assert args.image is not None, "Image must be provided for i2v"
        pipeline = wan.WanI2V(config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0)
        image = Image.open(args.image).convert("RGB")
        video = pipeline.generate(
            args.prompt,
            image,
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=(args.sample_shift or cfg.sample_shift),
            sampling_steps=(args.sample_steps or cfg.sample_steps),
            guide_scale=(args.sample_guide_scale or cfg.sample_guide_scale),
            seed=(args.base_seed if args.base_seed >= 0 else 42),
            offload_model=args.offload_model,
        )

    elif args.task.startswith("s2v"):
        assert args.image is not None, "Image must be provided for s2v"
        assert (args.audio or args.enable_tts), "Audio or enable_tts must be provided"
        pipeline = wan.WanS2V(config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0)
        video = pipeline.generate(
            input_prompt=args.prompt,
            ref_image_path=args.image,
            audio_path=args.audio,
            enable_tts=args.enable_tts,
            tts_prompt_audio=args.tts_prompt_audio,
            tts_prompt_text=args.tts_prompt_text,
            tts_text=args.tts_text,
            num_repeat=None,
            pose_video=None,
            max_area=MAX_AREA_CONFIGS[args.size],
            infer_frames=args.frame_num,
            shift=(args.sample_shift or cfg.sample_shift),
            sample_solver="unipc",
            sampling_steps=(args.sample_steps or cfg.sample_steps),
            guide_scale=(args.sample_guide_scale or cfg.sample_guide_scale),
            seed=(args.base_seed if args.base_seed >= 0 else 42),
            offload_model=args.offload_model,
            init_first_frame=False,
        )

    else:
        raise ValueError(f"Unsupported task: {args.task}")

    # Save output file
    if args.save_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_file = f"{args.task}_{timestamp}.mp4"

    logging.info(f"Saving video to {args.save_file}")
    save_video(
        tensor=video[None],
        save_file=args.save_file,
        fps=args.fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )

    # Merge audio for S2V
    if args.task.startswith("s2v"):
        logging.info("Merging audio into video...")
        merge_video_audio(
            video_path=args.save_file,
            audio_path=args.audio if not args.enable_tts else "tts.wav",
        )

    logging.info("Finished generation")
    return args.save_file


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
