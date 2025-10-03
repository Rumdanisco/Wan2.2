# handler.py
import runpod
import subprocess
import os
import uuid
import sys
import requests

# ✅ Lock to a single I2V model (Wan2.2-I2V-A8B)
WAN_MODEL = "Wan-AI/Wan2.2-I2V-A8B"

def generate_video(input_params):
    """
    Run generate.py with parameters from RunPod API request.
    Supports Image-to-Video (I2V).
    """

    prompt = input_params.get("prompt", "A futuristic city at night")
    size = input_params.get("size", "1280*720")
    steps = int(input_params.get("steps", 25))
    seed = int(input_params.get("seed", 42))

    # ✅ Image is required for I2V
    image_path = input_params.get("image")
    if not image_path:
        return {"error": "Task 'i2v' requires an image."}

    # ✅ Support image URLs (download to /workspace/)
    if image_path.startswith("http://") or image_path.startswith("https://"):
        local_image = f"/workspace/{uuid.uuid4().hex}.png"
        try:
            r = requests.get(image_path, timeout=30)
            r.raise_for_status()
            with open(local_image, "wb") as f:
                f.write(r.content)
            image_path = local_image
        except Exception as e:
            return {"error": f"Failed to download image from URL: {str(e)}"}

    # ✅ User type rules
    user_type = input_params.get("user_type", "free")

    # ✅ Prefer frame_num from frontend, fallback to duration
    frame_num = input_params.get("frame_num")
    if frame_num is not None:
        try:
            frame_num = int(frame_num)
        except ValueError:
            return {"error": "Invalid frame_num value."}
    else:
        duration = int(input_params.get("duration", 5))
        if user_type == "free" and duration > 5:
            return {"error": "Upgrade required for longer videos."}
        fps = 16
        frame_num = duration * fps

    output_file = f"/workspace/output_{uuid.uuid4().hex}.mp4"

    # ✅ Call generate.py
    cmd = [
        sys.executable, "generate.py",
        "--task", "i2v-A8B",
        "--prompt", prompt,
        "--image", image_path,
        "--size", size,
        "--frame_num", str(frame_num),
        "--sample_steps", str(steps),
        "--base_seed", str(seed),
        "--ckpt_dir", WAN_MODEL
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logs = result.stdout + "\n" + result.stderr
    except subprocess.CalledProcessError as e:
        return {
            "error": str(e),
            "logs": (e.stdout or "") + (e.stderr or "")
        }

    # ✅ Find the generated file
    for f in os.listdir("/workspace"):
        if f.endswith(".mp4"):
            output_file = os.path.join("/workspace", f)

    return {
        "video_path": output_file,
        "logs": logs,
        "user_type": user_type,
        "frames": frame_num,
        "task": "i2v-A8B",
        "repo": WAN_MODEL
    }

# RunPod entrypoint
runpod.serverless.start({"handler": generate_video})
