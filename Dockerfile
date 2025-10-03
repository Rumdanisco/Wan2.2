# Use NVIDIA PyTorch base image with CUDA and cuDNN
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /workspace

# Set environment variables for Hugging Face + Temp dirs
ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/workspace/cache/hf \
    TRANSFORMERS_CACHE=/workspace/cache/transformers \
    TMPDIR=/workspace/cache/tmp \
    HF_TOKEN=${HF_TOKEN} \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Create cache directories
RUN mkdir -p /workspace/cache/hf \
    && mkdir -p /workspace/cache/transformers \
    && mkdir -p /workspace/cache/tmp

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ffmpeg build-essential \
    libgl1 libglib2.0-0 libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# RunPod entrypoint
CMD ["python3", "handler.py"]
