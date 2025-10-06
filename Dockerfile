# ✅ Use official PyTorch base image with CUDA & cuDNN
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# Set the working directory inside the container
WORKDIR /workspace

# ✅ Install essential system packages
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
 && rm -rf /var/lib/apt/lists/*

# ✅ Copy your requirements file into container
COPY requirements.txt .

# ✅ Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ✅ Copy your handler file
COPY handler.py .

# ✅ (Optional) Create an output directory for saving generated videos
RUN mkdir -p /workspace/output

# ✅ Set environment variables for Hugging Face access and cache
# These can also be set in RunPod Environment Variables
ENV MODEL_REPO=Wan-AI/Wan2.2-TI2V-1.3B-Diffusers
ENV HF_HOME=/workspace/.cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface
ENV HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface

# ✅ Expose port (if RunPod needs it)
EXPOSE 8000

# ✅ Start the RunPod serverless handler
CMD ["python3", "handler.py"]
