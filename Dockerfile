FROM python:3.10-slim

# Install system dependencies required by OpenCV and Mediapipe
RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip

# Install mediapipe separately first
RUN pip install mediapipe==0.10.9

# Then install everything else
COPY requirements.txt .
RUN pip install -r requirements.txt

EXPOSE 5000

# Start Flask app
CMD ["python", "app.py"]
