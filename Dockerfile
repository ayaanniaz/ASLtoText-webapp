FROM python:3.10-slim

# System libraries needed for OpenCV & MediaPipe
RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 libgl1 libglib2.0-0 \
    libxrender1 libxfixes3 libxi6 libxrandr2 \
    libxss1 libgconf-2-4 libasound2 \
    && rm -rf /var/lib/apt/lists/*

# Prevent GPU/OpenGL usage for MediaPipe/TensorFlow
ENV CUDA_VISIBLE_DEVICES=""
ENV MPLBACKEND="Agg"
ENV GLOG_minloglevel=2

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# Copy the rest of the application
COPY . /app

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "120", "app:app"]
