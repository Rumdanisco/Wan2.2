import runpod
import uuid
import torch
from diffusers import DiffusionPipeline
import os
import requests
import shutil


def generate_video(job):
    """
    RunPod handler for Wan 2.1 T2V 1.3B Diffusers (text-to-video and image-to-video)
    """

    # 🧹 Clear Hugging Face cache before loading model
    shutil.rmtree("/root/.cache/huggingface", ignore_errors=True)
    os.makedirs("/root/.cache/huggingface", exist_ok=True)

    inputs = job["input"]
    prompt = inputs.get("prompt", "A cinematic shot of a futuristic city at night.")
    image_url = inputs.get("image", None)
    output_path = f"/workspace/output/output_{uuid.uuid4().hex}.mp4"

    # ✅ Load model repo and token
    model_repo = os.getenv("MODEL_REPO", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    token = os.getenv("HF_TOKEN")

    print(f"🚀 Loading model from {model_repo}")
    pipe = DiffusionPipeline.from_pretrained(
        model_repo,
        torch_dtype=torch.float16,
        token=token
    ).to("cuda")

    # ✅ If image is provided → Image-to-Video mode
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
            return {"error": f"Failed to download or process image: {str(e)}"}
    else:
        print(f"🎬 Generating text-to-video for: {prompt}")
        result = pipe(prompt=prompt)

    # ✅ Save generated video
    video = result.get("video") or result[0]
    video.save(output_path)

    print("✅ Video saved:", output_path)
    return {"video_path": output_path, "prompt": prompt}


# ✅ Start the RunPod handler
runpod.serverless.start({"handler": generate_video})
