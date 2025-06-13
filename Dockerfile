FROM python:3.12-slim

# System libraries needed for OpenCV & MediaPipe
RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Prevent GPU/OpenGL usage for MediaPipe/TensorFlow
ENV CUDA_VISIBLE_DEVICES=""
ENV MPLBACKEND="Agg"

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip setuptools wheel

# Install mediapipe
RUN pip install mediapipe==0.10.21

# Install CPU-only TensorFlow (lighter weight)
RUN pip install tensorflow-cpu==2.13.0

# Copy and install other dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

EXPOSE 5000
CMD ["python", "app.py"]
