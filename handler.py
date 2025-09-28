# handler.py
import runpod
import subprocess
import os
import uuid


def generate_video(input_params):
    """
    Run generate.py with parameters from RunPod API request.
    """

    task = input_params.get("task", "t2v")   # "t2v" or "i2v"
    prompt = input_params.get("prompt", "A cinematic scene of a dragon flying")
    size = input_params.get("size", "1280*720")
    steps = int(input_params.get("steps", 25))
    seed = int(input_params.get("seed", 42))

    # User type: free or pro
    user_type = input_params.get("user_type", "free")  # default = free

    # Duration request from frontend
    duration = int(input_params.get("duration", 5))  # 5 or 10 sec

    # Enforce limits: free users can only do 5 sec
    if user_type == "free" and duration > 5:
        return {"error": "Upgrade required for longer videos."}

    # Map duration (seconds) to frames, assuming ~16 FPS
    fps = 16
    frame_num = duration * fps

    # temp output filename
    output_file = f"/workspace/output_{uuid.uuid4().hex}.mp4"

    # Build base command
    cmd = [
        "python", "generate.py",
        "--task", task,
        "--prompt", prompt,
        "--size", size,
        "--frame_num", str(frame_num),
        "--steps", str(steps),
        "--seed", str(seed)
    ]

    # Add image if i2v
    if task == "i2v":
        image_path = input_params.get("image")
        if not image_path:
            return {"error": "Task 'i2v' requires an image"}
        cmd.extend(["--image", image_path])

    # Run the generator
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logs = result.stdout + "\n" + result.stderr
    except subprocess.CalledProcessError as e:
        return {"error": str(e), "logs": e.stdout + e.stderr}

    # Find the latest generated .mp4 in workspace
    for f in os.listdir("/workspace"):
        if f.endswith(".mp4"):
            output_file = os.path.join("/workspace", f)

    return {
        "video_path": output_file,
        "logs": logs,
        "user_type": user_type,
        "duration": duration,
        "frames": frame_num
    }


# Start RunPod serverless handler
runpod.serverless.start({"handler": generate_video})
