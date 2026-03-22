import os
import io
import base64
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.applications.efficientnet import preprocess_input
    from tensorflow.keras.preprocessing import image as keras_image
    HAS_TF = True
except ImportError:
    HAS_TF = False

# ─────────────────────────── app setup ───────────────────────────
app = FastAPI(title="Distracted Driver Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────── constants ───────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT      = os.path.dirname(BASE_DIR) if os.path.basename(BASE_DIR) == 'Backend' else BASE_DIR
FRONTEND_DIR   = os.path.join(REPO_ROOT, "Frontend")

# Serve chart PNGs from repo root at /static
app.mount("/static", StaticFiles(directory=REPO_ROOT), name="static")

# Serve frontend assets (CSS/JS) at /frontend
if os.path.isdir(FRONTEND_DIR):
    app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")

MODEL_PATH     = os.path.join(BASE_DIR, "best_model.h5")
SAMPLES_DIR    = os.path.join(BASE_DIR, "sample_images")
IMG_SIZE       = (300, 300)

@app.get("/")
def serve_index():
    """Serve the frontend index.html so everything runs on same origin."""
    path = os.path.join(FRONTEND_DIR, "index.html")
    if not os.path.exists(path):
        return {"detail": "Frontend not found. Open Frontend/index.html directly."}
    return FileResponse(path)


CLASS_META = [
    {"id": "c0", "label": "Safe Driving",              "severity": "safe",   "severity_score": 0, "color": "#22c55e", "description": "Driver is attentive, hands on wheel, focused on the road."},
    {"id": "c1", "label": "Texting — Right Hand",      "severity": "high",   "severity_score": 5, "color": "#ef4444", "description": "Driver is texting or using a phone with their right hand."},
    {"id": "c2", "label": "Phone Call — Right Hand",   "severity": "high",   "severity_score": 4, "color": "#f97316", "description": "Driver is talking on the phone held in their right hand."},
    {"id": "c3", "label": "Texting — Left Hand",       "severity": "high",   "severity_score": 5, "color": "#ef4444", "description": "Driver is texting or using a phone with their left hand."},
    {"id": "c4", "label": "Phone Call — Left Hand",    "severity": "high",   "severity_score": 4, "color": "#f97316", "description": "Driver is talking on the phone held in their left hand."},
    {"id": "c5", "label": "Operating Radio",           "severity": "medium", "severity_score": 3, "color": "#eab308", "description": "Driver is adjusting the stereo, radio, or other console control."},
    {"id": "c6", "label": "Drinking",                  "severity": "medium", "severity_score": 2, "color": "#eab308", "description": "Driver is drinking from a bottle or cup while driving."},
    {"id": "c7", "label": "Reaching Behind",           "severity": "medium", "severity_score": 3, "color": "#eab308", "description": "Driver is reaching behind or to the side, away from wheel."},
    {"id": "c8", "label": "Hair and Makeup",           "severity": "medium", "severity_score": 2, "color": "#eab308", "description": "Driver is attending to hair, makeup, or personal grooming."},
    {"id": "c9", "label": "Talking to Passenger",      "severity": "low",    "severity_score": 1, "color": "#3b82f6", "description": "Driver is conversing with a passenger inside the vehicle."},
]
CLASS_KEYS = [c["id"] for c in CLASS_META]

# ─────────────────────────── model ───────────────────────────────
model = None
model_loaded = False

@app.on_event("startup")
async def load():
    global model, model_loaded
    if HAS_TF and os.path.exists(MODEL_PATH):
        try:
            print("Loading EfficientNetB3 model …")
            model = load_model(MODEL_PATH)
            model_loaded = True
            print("Model ready ✓")
        except Exception as e:
            print(f"Model load error: {e}")
    else:
        print("Model not found — running in mock mode.")

# ─────────────────────────── helpers ─────────────────────────────
def img_to_array(img_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize(IMG_SIZE)
    arr = keras_image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    return preprocess_input(arr)

def mock_prediction():
    scores = np.random.dirichlet(np.ones(10), size=1)[0].tolist()
    return scores

# ─────────────────────────── routes ──────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model_loaded}

@app.get("/classes")
def get_classes():
    return {"classes": CLASS_META}

@app.get("/sample-images")
def list_samples():
    """Return metadata for all sample images available in SAMPLES_DIR."""
    if not os.path.exists(SAMPLES_DIR):
        return {"samples": []}
    
    samples = []
    for fname in sorted(os.listdir(SAMPLES_DIR)):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            samples.append({
                "filename": fname,
                "url": f"/sample-image/{fname}",
            })
    return {"samples": samples}

@app.get("/sample-image/{filename}")
def serve_sample(filename: str):
    """Serve a sample image file."""
    safe_name = os.path.basename(filename)
    path = os.path.join(SAMPLES_DIR, safe_name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path, media_type="image/jpeg")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Not an image.")
    
    content = await file.read()
    
    if not model_loaded:
        scores = mock_prediction()
        pred_idx = int(np.argmax(scores))
        return _build_response(scores, pred_idx, mocked=True)
    
    try:
        arr = img_to_array(content)
        preds = model.predict(arr, verbose=0)[0].tolist()
        pred_idx = int(np.argmax(preds))
        return _build_response(preds, pred_idx, mocked=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _build_response(scores: list, pred_idx: int, mocked: bool):
    meta = CLASS_META[pred_idx]
    all_scores = [
        {**CLASS_META[i], "confidence": round(scores[i] * 100, 2)}
        for i in range(10)
    ]
    # sort descending for easy frontend rendering
    all_scores_sorted = sorted(all_scores, key=lambda x: x["confidence"], reverse=True)
    return {
        "status": "success",
        "mocked": mocked,
        "prediction": meta["label"],
        "class_id": meta["id"],
        "severity": meta["severity"],
        "severity_score": meta["severity_score"],
        "color": meta["color"],
        "confidence": round(scores[pred_idx] * 100, 2),
        "all_scores": all_scores_sorted,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
