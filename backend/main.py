# backend/main.py
from pathlib import Path
import importlib
import traceback
import uuid
import time
from typing import Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io
import numpy as np
import threading

# create app first
app = FastAPI(title="SkinDiseaseAI - Inference API", version="0.2")

# ensure static directory exists so StaticFiles won't error at startup
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# mount static files (so /static/... serves files)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# dedicated favicon route (serves static/favicon.ico if present)
@app.get("/favicon.ico")
async def favicon():
    ico_path = STATIC_DIR / "favicon.ico"
    if ico_path.exists():
        return FileResponse(str(ico_path))
    # return 204 No Content if no favicon is provided
    return Response(status_code=204)


def _load_get_model():
    """
    Try to import a get_model() factory from backend.model_loader or model_loader.
    If import fails, return a fallback get_model() that provides a DummyModel.
    """
    # Try top-level module import (when running from backend/)
    try:
        mod = importlib.import_module("model_loader")
        return mod.get_model
    except Exception:
        pass

    # Try package import (when running as package: backend.main)
    try:
        mod = importlib.import_module("backend.model_loader")
        return mod.get_model
    except Exception:
        pass

    # Fallback dummy implementation
    class DummyModel:
        def __init__(self):
            self.framework = "dummy"
            self.model = None

        def predict(self, img_array: np.ndarray):
            return [
                {"label": "benign", "confidence": 0.78},
                {"label": "malignant", "confidence": 0.12},
                {"label": "other", "confidence": 0.10},
            ]

        def predict_image(self, pil_img):
            # reuse predict to match ModelWrapper interface
            arr = np.array(pil_img.resize((224, 224))).astype("float32") / 255.0
            return self.predict(arr)

    def _get_model_fallback():
        return DummyModel()

    # print import traceback for debugging, but do not crash
    print("Warning: model_loader not found; using DummyModel fallback.")
    return _get_model_fallback


# Create/get model singleton
_get_model_factory = _load_get_model()
_model_instance = None
_model_lock = threading.Lock()  # protects _model_instance during reloads


def get_model():
    global _model_instance
    if _model_instance is None:
        _model_instance = _get_model_factory()
    return _model_instance


@app.get("/")
def root():
    return {"status": "ok", "service": "SkinDiseaseAI Inference API"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an image file (jpg/png). Returns model predictions.
    Uses model_loader.get_model() if available; otherwise returns dummy predictions.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Unable to read image")

    model = get_model()

    # prefer method name predict_image for ModelWrapper; fall back to predict
    try:
        if hasattr(model, "predict_image"):
            preds = model.predict_image(img)
        elif hasattr(model, "predict"):
            # if predict expects numpy array
            arr = np.array(img.resize((224, 224))).astype("float32") / 255.0
            preds = model.predict(arr)
        else:
            raise RuntimeError("Loaded model has no predictable interface")
    except Exception:
        # If inference fails, return an error with a safe message and include debug info in logs
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Model inference failed")

    response = {
        "predictions": preds,
        "metadata": {
            "model_framework": getattr(model, "framework", "unknown"),
            "model_loaded": getattr(model, "model", None) is not None,
        },
    }
    return JSONResponse(content=response)


# -------------------
# Admin endpoints (non-blocking reload)
# -------------------

@app.get("/admin/model_status")
def admin_model_status():
    """
    Returns basic status about the currently loaded model.
    """
    try:
        model = get_model()
        return {
            "loaded": getattr(model, "model", None) is not None,
            "framework": getattr(model, "framework", "unknown"),
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# simple in-memory job store (thread-safe) for reload jobs
_reload_jobs_lock = threading.Lock()
_reload_jobs: Dict[str, Dict[str, Any]] = {}  # job_id -> {status, started_at, finished_at, result, error}


def _run_reload_job(job_id: str):
    """Background worker that attempts to reload the model and records result."""
    with _reload_jobs_lock:
        _reload_jobs[job_id]["status"] = "running"
        _reload_jobs[job_id]["started_at"] = time.time()

    try:
        # Attempt to reload the model_loader module (both import styles)
        reloaded = False
        try:
            import model_loader as ml_mod
            importlib.reload(ml_mod)
            reloaded = True
        except Exception:
            try:
                import backend.model_loader as ml_pkg
                importlib.reload(ml_pkg)
                reloaded = True
            except Exception:
                pass

        # Drop cached instance and re-create factory, protected by model lock
        global _model_instance, _get_model_factory
        with _model_lock:
            _model_instance = None
            _get_model_factory = _load_get_model()
            # Instantiate the model (may take time)
            model = get_model()

        result = {
            "model_framework": getattr(model, "framework", "unknown"),
            "model_loaded": getattr(model, "model", None) is not None,
            "reloaded_module": reloaded,
        }

        with _reload_jobs_lock:
            _reload_jobs[job_id].update({
                "status": "success",
                "finished_at": time.time(),
                "result": result,
                "error": None,
            })
    except Exception as exc:
        traceback.print_exc()
        with _reload_jobs_lock:
            _reload_jobs[job_id].update({
                "status": "error",
                "finished_at": time.time(),
                "result": None,
                "error": str(exc),
            })


@app.post("/admin/reload_model")
def admin_reload_model_nonblocking():
    """
    Starts a background job to reload the model.
    Returns a job_id immediately; check /admin/reload_status/{job_id} for progress/result.
    """
    job_id = str(uuid.uuid4())
    with _reload_jobs_lock:
        _reload_jobs[job_id] = {
            "status": "pending",
            "started_at": None,
            "finished_at": None,
            "result": None,
            "error": None,
        }

    # start the background thread
    t = threading.Thread(target=_run_reload_job, args=(job_id,), daemon=True)
    t.start()

    return {"status": "accepted", "job_id": job_id}


@app.get("/admin/reload_status/{job_id}")
def admin_reload_status(job_id: str):
    with _reload_jobs_lock:
        job = _reload_jobs.get(job_id)
        if not job:
            return JSONResponse(status_code=404, content={"error": "job not found"})
        return {
            "job_id": job_id,
            "status": job["status"],
            "started_at": job["started_at"],
            "finished_at": job["finished_at"],
            "result": job["result"],
            "error": job["error"],
        }
