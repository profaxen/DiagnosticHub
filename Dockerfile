# Hugging Face Spaces — Docker deployment for PneumoScan Pro
FROM python:3.11-slim

# System dependencies for OpenCV + TensorFlow + wget
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
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

# Install Python dependencies first (cached layer)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gdown

# Copy all project files
COPY --chown=user . .

# Create necessary directories
RUN mkdir -p models outputs/plots outputs/confusion_matrices outputs/reports static/css static/js templates

# ── Download trained models during build (baked into image) ──────────────────
# Using gdown with --fuzzy to handle Google Drive large-file confirm page
# Advanced model: ResNet50V2 (253 MB)
RUN gdown --fuzzy "https://drive.google.com/uc?id=1u4WaQNN-tHTAqJsOkkjAyhwrstBrfm83" \
    -O models/advanced_best.h5 || \
    wget -q --show-progress \
    "https://drive.usercontent.google.com/download?id=1u4WaQNN-tHTAqJsOkkjAyhwrstBrfm83&export=download&confirm=t" \
    -O models/advanced_best.h5

# Baseline model: CNN (128 MB)
RUN gdown --fuzzy "https://drive.google.com/uc?id=1AXS9Z-RRVEl2tP4ZORw9RnKUM0hV2zKK" \
    -O models/baseline_best.h5 || \
    wget -q --show-progress \
    "https://drive.usercontent.google.com/download?id=1AXS9Z-RRVEl2tP4ZORw9RnKUM0hV2zKK&export=download&confirm=t" \
    -O models/baseline_best.h5

# Verify models were downloaded correctly (fail build if missing)
RUN python -c "
import os, sys
for name in ['advanced_best.h5', 'baseline_best.h5']:
    path = f'models/{name}'
    if not os.path.exists(path):
        print(f'ERROR: {path} not found!', file=sys.stderr)
        sys.exit(1)
    size = os.path.getsize(path) / 1024 / 1024
    if size < 10:
        print(f'ERROR: {path} is only {size:.1f} MB — likely a corrupt download!', file=sys.stderr)
        sys.exit(1)
    print(f'OK: {name} — {size:.1f} MB')
"

# HF Spaces uses port 7860
EXPOSE 7860

# Start FastAPI via uvicorn on port 7860
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
