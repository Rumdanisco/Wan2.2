# ✅ Use a stable PyTorch image with CUDA support
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /workspace

# ✅ Install essential system packages
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
 && rm -rf /var/lib/apt/lists/*

# ✅ Copy requirements first (for layer caching)
COPY requirements.txt .

# ✅ Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ✅ Copy handler file
COPY handler.py .

# ✅ Create output folder
RUN mkdir -p /workspace/output

# ✅ Environment Variables
ENV MODEL_REPO=Wan-AI/Wan2.1-T2V-1.3B-Diffusers
ENV HF_HOME=/workspace/.cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface
ENV DIFFUSERS_CACHE=/workspace/.cache/huggingface

# ✅ Expose port (if needed)
EXPOSE 8000

# ✅ Default command
CMD ["python3", "handler.py"]
