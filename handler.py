import runpod
import uuid
import torch
from diffusers import DiffusionPipeline
import os
import requests


def generate_video(job):
    """
    RunPod handler for Wan 2.2 TI2V-5B Diffusers (text-to-video and image-to-video)
    """
    inputs = job["input"]
    prompt = inputs.get("prompt", "A cinematic shot of a futuristic city at night.")
    image_url = inputs.get("image", None)
    output_path = f"/workspace/output_{uuid.uuid4().hex}.mp4"

    # Model info from environment variables
    model_repo = os.getenv("MODEL_REPO", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    token = os.getenv("HF_TOKEN")

    print(f"🚀 Loading model from {model_repo}")
    pipe = DiffusionPipeline.from_pretrained(
        model_repo,
        torch_dtype=torch.float16,
        token=token
    ).to("cuda")

    # If input image provided → image-to-video
    if image_url:
        img_path = f"/workspace/input_{uuid.uuid4().hex}.png"
        try:
            r = requests.get(image_url, timeout=30)
            r.raise_for_status()
            with open(img_path, "wb") as f:
                f.write(r.content)
            print(f"📥 Downloaded input image: {img_path}")
            result = pipe(prompt=prompt, image=img_path)
        except Exception as e:
            return {"error": f"Failed to download image: {str(e)}"}
    else:
        print(f"🎬 Generating text-to-video for: {prompt}")
        result = pipe(prompt=prompt)

    # Save output video
    video = result.get("video") or result[0]
    video.save(output_path)

    print("✅ Video saved:", output_path)
    return {"video_path": output_path, "prompt": prompt}


runpod.serverless.start({"handler": generate_video})
