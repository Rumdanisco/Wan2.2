# handler.py
import runpod
import os
import tempfile
import shutil
import sys
import argparse
from generate import _parse_args, generate


def run_inference(job):
    """
    RunPod handler for Wan2.2.
    job['input'] should contain:
        task: "t2v-A14B" | "i2v-A14B" | "animate-14B" | "s2v-14B"
        prompt: text prompt
        image: (optional) path or uploaded file
        audio: (optional) path for s2v
        enable_tts: (optional) bool
        size: (optional) resolution, default "1280*720"
        frame_num: (optional) frames
    """

    inputs = job["input"]

    # temp working dir
    work_dir = tempfile.mkdtemp()
    output_path = os.path.join(work_dir, "output.mp4")

    # build CLI-style args
    cli_args = [
        "--task", inputs.get("task", "t2v-A14B"),
        "--ckpt_dir", inputs.get("ckpt_dir", "./Wan2.2-T2V-A14B"),
        "--prompt", inputs.get("prompt", "A cat riding a surfboard"),
        "--size", inputs.get("size", "1280*720"),
        "--save_file", output_path
    ]

    if "image" in inputs and inputs["image"]:
        cli_args += ["--image", inputs["image"]]

    if "audio" in inputs and inputs["audio"]:
        cli_args += ["--audio", inputs["audio"]]

    if inputs.get("enable_tts", False):
        cli_args += ["--enable_tts"]

    if "frame_num" in inputs:
        cli_args += ["--frame_num", str(inputs["frame_num"])]

    # parse args using original _parse_args
    parser = _parse_args()
    args = parser.parse_args(cli_args)

    # call original generator
    generate(args)

    # move result to /runpod-volume
    result_path = f"/runpod-volume/{os.path.basename(output_path)}"
    shutil.copy(output_path, result_path)

    return {"output": result_path}


# start RunPod serverless handler
runpod.serverless.start({"handler": run_inference})
