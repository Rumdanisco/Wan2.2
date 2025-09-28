# Use NVIDIA CUDA base image with PyTorch
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /workspace

# System dependencies
RUN apt-get update && apt-get install -y \
    git wget curl python3 python3-pip ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# Copy project files
COPY . .

# Expose RunPod port
EXPOSE 8000

# Default command (RunPod will override with handler)
CMD ["python3", "-u", "handler.py"]
