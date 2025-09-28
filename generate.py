# generate.py
# Rewritten for RunPod compatibility, keeping all Wan2.2 features.
# Based on the original Wan 2.2 generate.py

import argparse
import logging
import os
import sys
import warnings
from datetime import datetime
import random

warnings.filterwarnings("ignore")

import torch
import torch.distributed as dist
from PIL import Image

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.distributed.util import init_distributed_group
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import merge_video_audio, save_video, str2bool


# Example prompts for different tasks
EXAMPLE_PROMPT = {
    "t2v-A14B": {"prompt": "A cinematic shot of a dragon flying over snowy mountains."},
    "i2v-A14B": {
        "prompt": "A cat stands in a beach scene.",
        "image": "examples/cat.png",
    },
    "animate-14B": {
        "prompt": "视频中的人在做动作",
        "video": "",
        "pose": "",
        "mask": "",
    },
    "s2v-14B": {
        "prompt": "A singer is singing on stage.",
        "image": "examples/cat.png",
        "audio": "examples/talk.wav",
        "tts_prompt_audio": "examples/zero_shot_prompt.wav",
        "tts_prompt_text": "Hello from Wan2.2!",
        "tts_text": "This is a synthesized voice test.",
    },
}


def _validate_args(args):
    """Basic validation and default filling."""
    assert args.ckpt_dir is not None, "Please specify --ckpt_dir (checkpoint dir)."
    assert args.task in WAN_CONFIGS, f"Unsupported task: {args.task}"

    # Fill defaults from EXAMPLE_PROMPT if missing
    if args.prompt is None and args.task in EXAMPLE_PROMPT:
        args.prompt = EXAMPLE_PROMPT[args.task].get("prompt")
    if args.image is None and "image" in EXAMPLE_PROMPT.get(args.task, {}):
        args.image = EXAMPLE_PROMPT[args.task]["image"]
    if args.audio is None and args.task.startswith("s2v"):
        args.audio = EXAMPLE_PROMPT[args.task].get("audio")
    if args.enable_tts and (
        args.tts_prompt_audio is None or args.tts_text is None
    ):
        args.tts_prompt_audio = EXAMPLE_PROMPT[args.task].get("tts_prompt_audio")
        args.tts_prompt_text = EXAMPLE_PROMPT[args.task].get("tts_prompt_text")
        args.tts_text = EXAMPLE_PROMPT[args.task].get("tts_text")

    # Size check
    if not args.task.startswith("s2v"):
        assert args.size in SUPPORTED_SIZES[args.task], (
            f"Unsupported size {args.size} for task {args.task}. "
            f"Valid sizes: {', '.join(SUPPORTED_SIZES[args.task])}"
        )

    # Fill default sampling params from config
    cfg = WAN_CONFIGS[args.task]
    if args.sample_steps is None:
        args.sample_steps = cfg.sample_steps
    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift
    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale
    if args.frame_num is None:
        args.frame_num = cfg.frame_num

    # Seed
    args.base_seed = (
        args.base_seed if args.base_seed >= 0 else random.randint(0, sys.maxsize)
    )


def _parse_args():
    """Parse CLI arguments (also used in handler)."""
    parser = argparse.ArgumentParser(
        description="Wan2.2 Multi-Task Video Generation (T2V, I2V, Animate, S2V)"
    )
    parser.add_argument("--task", type=str, default="t2v-A14B", choices=list(WAN_CONFIGS.keys()))
    parser.add_argument("--size", type=str, default="1280*720", choices=list(SIZE_CONFIGS.keys()))
    parser.add_argument("--frame_num", type=int, default=None)
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--offload_model", type=str2bool, default=True)
    parser.add_argument("--ulysses_size", type=int, default=1)
    parser.add_argument("--t5_fsdp", action="store_true", default=False)
    parser.add_argument("--t5_cpu", action="store_true", default=False)
    parser.add_argument("--dit_fsdp", action="store_true", default=False)
    parser.add_argument("--save_file", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--use_prompt_extend", action="store_true", default=False)
    parser.add_argument("--prompt_extend_method", type=str, default="local_qwen", choices=["dashscope", "local_qwen"])
    parser.add_argument("--prompt_extend_model", type=str, default=None)
    parser.add_argument("--prompt_extend_target_lang", type=str, default="zh", choices=["zh", "en"])
    parser.add_argument("--base_seed", type=int, default=-1)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++"])
    parser.add_argument("--sample_steps", type=int, default=None)
    parser.add_argument("--sample_shift", type=float, default=None)
    parser.add_argument("--sample_guide_scale", type=float, default=None)
    parser.add_argument("--convert_model_dtype", action="store_true", default=False)

    # Animate
    parser.add_argument("--src_root_path", type=str, default=None)
    parser.add_argument("--refert_num", type=int, default=77)
    parser.add_argument("--replace_flag", action="store_true", default=False)
    parser.add_argument("--use_relighting_lora", action="store_true", default=False)

    # S2V (speech-to-video)
    parser.add_argument("--num_clip", type=int, default=None)
    parser.add_argument("--audio", type=str, default=None)
    parser.add_argument("--enable_tts", action="store_true", default=False)
    parser.add_argument("--tts_prompt_audio", type=str, default=None)
    parser.add_argument("--tts_prompt_text", type=str, default=None)
    parser.add_argument("--tts_text", type=str, default=None)
    parser.add_argument("--pose_video", type=str, default=None)
    parser.add_argument("--start_from_ref", action="store_true", default=False)
    parser.add_argument("--infer_frames", type=int, default=80)

    args = parser.parse_args()
    _validate_args(args)
    return args


def generate(args):
    """Main generation logic."""
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank

    # Setup logging
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.ERROR,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    cfg = WAN_CONFIGS[args.task]
    logging.info(f"Starting generation task={args.task} with config={cfg}")

    # Load image if provided
    img = None
    if args.image:
        img = Image.open(args.image).convert("RGB")

    # Prompt extend
    if args.use_prompt_extend:
        logging.info("Extending prompt...")
        if args.prompt_extend_method == "dashscope":
            expander = DashScopePromptExpander(
                model_name=args.prompt_extend_model, task=args.task, is_vl=img is not None
            )
        else:
            expander = QwenPromptExpander(
                model_name=args.prompt_extend_model,
                task=args.task,
                is_vl=img is not None,
                device=rank,
            )
        result = expander(args.prompt, image=img, tar_lang=args.prompt_extend_target_lang, seed=args.base_seed)
        args.prompt = result.prompt if result.status else args.prompt

    # Select pipeline
    if args.task.startswith("t2v"):
        pipeline = wan.WanT2V(config=cfg, checkpoint_dir=args.ckpt_dir, device_id=device)
        video = pipeline.generate(
            args.prompt, size=SIZE_CONFIGS[args.size],
            frame_num=args.frame_num, shift=args.sample_shift,
            sample_solver=args.sample_solver, sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale, seed=args.base_seed,
            offload_model=args.offload_model
        )
    elif args.task.startswith("i2v"):
        pipeline = wan.WanI2V(config=cfg, checkpoint_dir=args.ckpt_dir, device_id=device)
        video = pipeline.generate(
            args.prompt, img,
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num, shift=args.sample_shift,
            sample_solver=args.sample_solver, sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale, seed=args.base_seed,
            offload_model=args.offload_model
        )
    elif args.task.startswith("animate"):
        pipeline = wan.WanAnimate(config=cfg, checkpoint_dir=args.ckpt_dir, device_id=device)
        video = pipeline.generate(
            src_root_path=args.src_root_path, replace_flag=args.replace_flag,
            refert_num=args.refert_num, clip_len=args.frame_num,
            shift=args.sample_shift, sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps, guide_scale=args.sample_guide_scale,
            seed=args.base_seed, offload_model=args.offload_model
        )
    elif args.task.startswith("s2v"):
        pipeline = wan.WanS2V(config=cfg, checkpoint_dir=args.ckpt_dir, device_id=device)
        video = pipeline.generate(
            input_prompt=args.prompt, ref_image_path=args.image,
            audio_path=args.audio, enable_tts=args.enable_tts,
            tts_prompt_audio=args.tts_prompt_audio, tts_prompt_text=args.tts_prompt_text,
            tts_text=args.tts_text, num_repeat=args.num_clip, pose_video=args.pose_video,
            max_area=MAX_AREA_CONFIGS[args.size], infer_frames=args.infer_frames,
            shift=args.sample_shift, sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps, guide_scale=args.sample_guide_scale,
            seed=args.base_seed, offload_model=args.offload_model,
            init_first_frame=args.start_from_ref
        )
    else:
        raise ValueError(f"Unsupported task: {args.task}")

    # Save output file
    if not args.save_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_file = f"{args.task}_{args.size}_{timestamp}.mp4"

    save_video(video[None], args.save_file, fps=cfg.sample_fps, nrow=1, normalize=True, value_range=(-1, 1))

    # Merge audio if s2v
    if args.task.startswith("s2v"):
        merge_video_audio(args.save_file, args.audio if not args.enable_tts else "tts.wav")

    logging.info(f"Saved result to {args.save_file}")
    return args.save_file


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
