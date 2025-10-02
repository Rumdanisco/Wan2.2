# handler.py
import runpod
import subprocess
import os
import uuid
import sys

# ✅ Map WAN tasks to Hugging Face repos (correct repos)
WAN_MODELS = {
    "t2v-A14B": "Wan-AI/Wan2.2-T2V-A14B",   # Text-to-Video 14B
    "i2v-A14B": "Wan-AI/Wan2.2-I2V-A14B",   # Image-to-Video 14B
    "ti2v-5B": "Wan-AI/Wan-5B-TI2V",        # Text+Image-to-Video 5B
    "animate-14B": "Wan-AI/Wan2.2-T2V-A14B", # Using same 14B repo
    "s2v-14B": "Wan-AI/Wan2.2-T2V-A14B"      # Placeholder (speech-to-video if exists)
}

def generate_video(input_params):
    """
    Run generate.py with parameters from RunPod API request.
    """

    # Default = text-to-video
    raw_task = input_params.get("task", "t2v")
    task_map = {
        "t2v": "t2v-A14B",
        "i2v": "i2v-A14B",
        "ti2v": "ti2v-5B",
        "animate": "animate-14B",
        "s2v": "s2v-14B"
    }
    task = task_map.get(raw_task, raw_task)

    if task not in WAN_MODELS:
        return {"error": f"Unknown task: {task}"}

    repo = WAN_MODELS[task]  # Hugging Face repo for this task

    prompt = input_params.get("prompt", "A cinematic scene of a dragon flying")
    size = input_params.get("size", "1280*720")
    steps = int(input_params.get("steps", 25))
    seed = int(input_params.get("seed", 42))

    # User type
    user_type = input_params.get("user_type", "free")
    duration = int(input_params.get("duration", 5))

    if user_type == "free" and duration > 5:
        return {"error": "Upgrade required for longer videos."}

    fps = 16
    frame_num = duration * fps

    output_file = f"/workspace/output_{uuid.uuid4().hex}.mp4"

    # ✅ Pass ckpt_dir = Hugging Face repo
    cmd = [
        sys.executable, "generate.py",
        "--task", task,
        "--prompt", prompt,
        "--size", size,
        "--frame_num", str(frame_num),
        "--steps", str(steps),
        "--seed", str(seed),
        "--ckpt_dir", repo
    ]

    # Add image if i2v or ti2v
    if task in ["i2v-A14B", "ti2v-5B"]:
        image_path = input_params.get("image")
        if not image_path:
            return {"error": f"Task '{task}' requires an image"}
        cmd.extend(["--image", image_path])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logs = result.stdout + "\n" + result.stderr
    except subprocess.CalledProcessError as e:
        return {"error": str(e), "logs": (e.stdout or "") + (e.stderr or "")}

    for f in os.listdir("/workspace"):
        if f.endswith(".mp4"):
            output_file = os.path.join("/workspace", f)

    return {
        "video_path": output_file,
        "logs": logs,
        "user_type": user_type,
        "duration": duration,
        "frames": frame_num,
        "task": task,
        "repo": repo
    }

runpod.serverless.start({"handler": generate_video})
