# handler.py
import runpo
import subprocess
import os
import uuid
import sys

# ✅ Map WAN tasks to Hugging Face repos
WAN_MODELS = {
    "t2v-A14B": "Wan-AI/Wan2.2-T2V-A14B",   # Hugging Face repo for T2V 14B
    "i2v-A14B": "Wan-AI/Wan2.2-T2V-A14B",   # Same repo, different mode
    "ti2v-5B": "Wan-AI/Wan2.2-TI2V-5B",     # Example repo for 5B
    "animate-14B": "Wan-AI/Wan2.2-T2V-A14B",
    "s2v-14B": "Wan-AI/Wan2.2-S2V-14B"
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

    # ✅ Correct args: use sample_steps and base_seed
    cmd = [
        sys.executable, "generate.py",
        "--task", task,
        "--prompt", prompt,
        "--size", size,
        "--frame_num", str(frame_num),
        "--sample_steps", str(steps),
        "--base_seed", str(seed),
        "--ckpt_dir", repo
    ]

    # Add image if i2v
    if task == "i2v-A14B":
        image_path = input_params.get("image")
        if not image_path:
            return {"error": "Task 'i2v' requires an image"}
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
