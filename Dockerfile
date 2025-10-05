FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

RUN apt-get update && apt-get install -y git ffmpeg build-essential

COPY . .

RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

CMD ["python3", "handler.py"]
