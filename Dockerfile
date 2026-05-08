# Hugging Face Spaces — Docker deployment for PneumoScan Pro
FROM python:3.11-slim

# System dependencies for OpenCV + TensorFlow
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces runs as user 1000
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR $HOME/app

# Install Python dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY --chown=user . .

# Create necessary directories
RUN mkdir -p models outputs/plots outputs/confusion_matrices outputs/reports static/css static/js templates

# HF Spaces uses port 7860
EXPOSE 7860

# Start FastAPI via uvicorn on port 7860
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
