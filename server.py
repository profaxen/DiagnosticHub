"""
PneumoScan Pro — FastAPI Backend
Replaces Streamlit with a proper REST API + SPA architecture.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io
import os
import sys
import datetime
import base64
from src.predict import load_and_preprocess_image
import tempfile

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="PneumoScan Pro", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
os.makedirs("static/css", exist_ok=True)
os.makedirs("static/js", exist_ok=True)
os.makedirs("templates", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Model Check ───────────────────────────────────────────────────────────────
def check_models():
    """Verify model files exist — they are bundled in the repo via Git LFS."""
    os.makedirs("models", exist_ok=True)
    missing = []
    for name in ["advanced_best.h5", "baseline_best.h5"]:
        path = os.path.join("models", name)
        if not os.path.exists(path):
            missing.append(name)
        else:
            size_mb = os.path.getsize(path) / 1024 / 1024
            print(f"[OK] {name} — {size_mb:.1f} MB")
    if missing:
        print(f"[WARNING] Missing model files: {missing}", file=sys.stderr)
        print("[WARNING] Predictions will fail until models are present.", file=sys.stderr)

# ── Model Cache ───────────────────────────────────────────────────────────────
_model_cache: dict = {}

def get_model(engine: str):
    """Load model once and cache it."""
    if engine not in _model_cache:
        if engine == "pro":
            _model_cache[engine] = tf.keras.models.load_model("models/advanced_best.h5")
        else:
            _model_cache[engine] = tf.keras.models.load_model("models/baseline_best.h5")
        # Warm up
        _model_cache[engine](np.zeros((1, 224, 224, 3)))
        print(f"Model [{engine}] loaded and warmed up.")
    return _model_cache[engine]

# ── Grad-CAM ──────────────────────────────────────────────────────────────────
def compute_gradcam(model, img_array):
    try:
        target_model = model
        if "resnet50v2" in [l.name for l in model.layers]:
            target_model = model.get_layer("resnet50v2")

        last_conv = None
        for layer in reversed(target_model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv = layer
                break

        if not last_conv:
            return None

        grad_model = tf.keras.models.Model(
            inputs=[target_model.input],
            outputs=[last_conv.output, target_model.output],
        )

        with tf.GradientTape() as tape:
            conv_outs, preds = grad_model(img_array)
            loss = preds[:, 0]

        grads = tape.gradient(loss, conv_outs)[0]
        pooled = tf.reduce_mean(grads, axis=(0, 1))
        conv_outs = conv_outs[0]
        heatmap = conv_outs @ pooled[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        return heatmap.numpy()
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        return None

def image_to_base64(arr: np.ndarray) -> str:
    success, buf = cv2.imencode(".jpg", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf.tobytes()).decode()

# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    check_models()

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/predict")
async def predict(file: UploadFile = File(...), engine: str = "pro"):
    """Run pneumonia prediction on uploaded X-ray image."""
    if engine not in ("pro", "base"):
        raise HTTPException(400, "engine must be 'pro' or 'base'")

    # Read image bytes
    contents = await file.read()
    img_pil = Image.open(io.BytesIO(contents)).convert("RGB")

    # Save to temp file for preprocessing
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
        img_pil.save(tmp_path)

    try:
        model = get_model(engine)
        processed = load_and_preprocess_image(tmp_path)
        score = float(model.predict(processed, verbose=0)[0][0])

        # Compute Grad-CAM overlay
        orig = np.array(img_pil.resize((500, 500)))
        heatmap = compute_gradcam(model, processed)
        overlay_b64 = None

        if heatmap is not None:
            hm_resized = cv2.resize(heatmap, (500, 500))
            hm_colored = cv2.applyColorMap(np.uint8(255 * hm_resized), cv2.COLORMAP_JET)
            hm_rgb = cv2.cvtColor(hm_colored, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(orig, 0.6, hm_rgb, 0.4, 0)
            overlay_b64 = image_to_base64(overlay)

        # Original image base64
        orig_b64 = image_to_base64(orig)

        risk = "CRITICAL" if score > 0.7 else "MODERATE" if score > 0.4 else "LOW"
        confidence = score if score > 0.5 else 1 - score
        diagnosis = "PNEUMONIA DETECTED" if score > 0.5 else "LUNGS CLEAR"

        return JSONResponse({
            "score": score,
            "confidence": round(confidence * 100, 1),
            "diagnosis": diagnosis,
            "risk": risk,
            "positive": score > 0.5,
            "original_image": orig_b64,
            "heatmap_image": overlay_b64,
            "report_id": f"PS-{datetime.datetime.now().strftime('%y%m%d%H%M%S')}",
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
    finally:
        os.unlink(tmp_path)

@app.get("/api/report")
async def get_report(score: float, confidence: float, risk: str, report_id: str):
    """Generate downloadable clinical report."""
    status = "POSITIVE" if score > 0.5 else "NEGATIVE"
    diagnosis = "Pneumonia Detected" if score > 0.5 else "Normal Lung Findings"

    report = f"""
===========================================================
             PNEUMOSCAN PRO: CLINICAL REPORT
===========================================================
GENERATED ON: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
REPORT ID:    {report_id}
-----------------------------------------------------------

[SUMMARY OF FINDINGS]
---------------------
PATIENT STATUS:    {status}
PRIMARY DIAGNOSIS: {diagnosis}
AI CONFIDENCE:     {confidence:.1f}%
RISK CATEGORY:     {risk}

[CLINICAL INTERPRETATION]
-------------------------
The AI analysis detected radiographic markers consistent with
{'pulmonary consolidation and opacities typical of Pneumonia.' if score > 0.5 else 'clear lung fields with no significant pathological markers.'}
The Neural Attention Mapping (Grad-CAM) should be reviewed to
confirm the anatomical location of these findings.

[TECHNICAL SPECIFICATIONS]
--------------------------
Neural Backbone:   ResNet50V2 (Transfer Learning)
Engine Version:    v3.0.0
Architect:         ADARSH

-----------------------------------------------------------
DISCLAIMER: This is an AI-generated decision support report.
Final clinical diagnosis must be verified by a certified
radiologist. This report is for research and educational
purposes only.
===========================================================
    """.strip()

    return PlainTextResponse(
        content=report,
        headers={"Content-Disposition": f'attachment; filename="Report_{report_id}.txt"'},
    )

@app.get("/api/analytics")
async def analytics():
    """Return model analytics data."""
    metrics = {
        "baseline": {"accuracy": 90.0, "precision": 91.0, "recall": 93.0, "f1": 92.0},
        "advanced": {"accuracy": 93.0, "precision": 92.0, "recall": 98.0, "f1": 95.0},
        "dataset": {"normal": 1341, "pneumonia": 3875, "total": 5216},
        "plots": {
            "advanced_history": os.path.exists("outputs/plots/advanced_history.png"),
            "baseline_history": os.path.exists("outputs/plots/baseline_history.png"),
            "advanced_cm": os.path.exists("outputs/confusion_matrices/advanced_cm.png"),
            "baseline_cm": os.path.exists("outputs/confusion_matrices/baseline_cm.png"),
        }
    }
    return JSONResponse(metrics)

@app.get("/outputs/{path:path}")
async def serve_output(path: str):
    full = os.path.join("outputs", path)
    if os.path.exists(full):
        return FileResponse(full)
    raise HTTPException(404, "File not found")

# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
