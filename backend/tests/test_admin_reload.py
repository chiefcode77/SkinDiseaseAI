# backend/tests/test_admin_reload.py
import sys
import types
import time
import io
from PIL import Image
from fastapi.testclient import TestClient

# --- create and insert fake model_loader BEFORE importing main ---
fake_mod = types.ModuleType("model_loader")

class FakeModel:
    def __init__(self):
        self.framework = "fake"
        # non-None .model indicates "model_loaded" in the API
        self.model = object()

    def predict_image(self, pil_img):
        # simple deterministic response: a single-class response
        return [{"label": "fake", "confidence": 1.0}]

def get_model():
    return FakeModel()

fake_mod.get_model = get_model

# Insert the fake module so importlib.import_module("model_loader") / reload will find it
sys.modules["model_loader"] = fake_mod

# Now import main (after the fake is in place)
import main  # imports the app and the reload endpoints

def make_test_jpeg_bytes():
    # create a tiny in-memory JPEG for upload
    img = Image.new("RGB", (32, 32), color=(123, 50, 20))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()

def poll_reload_job(client: TestClient, job_id: str, timeout_seconds: float = 10.0):
    """Poll reload_status until job finishes; returns last_response dict."""
    poll_interval = 0.1
    start = time.time()
    last_response = None
    while time.time() - start < timeout_seconds:
        r = client.get(f"/admin/reload_status/{job_id}")
        assert r.status_code == 200
        last_response = r.json()
        status = last_response.get("status")
        if status not in ("pending", "running"):
            return last_response
        time.sleep(poll_interval)
    return last_response

def test_reload_job_with_mocked_model_loader_and_predict():
    """
    Inserted fake 'model_loader' before importing main so the background reload thread
    will find it quickly. Start the non-blocking reload endpoint and poll until finished.
    Then force the app to use FakeModel for predict and POST an in-memory JPEG asserting fake response.
    """
    client = TestClient(main.app)

    # Start reload job (non-blocking)
    r = client.post("/admin/reload_model")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "accepted"
    job_id = data["job_id"]

    # Wait until reload job finishes
    last_response = poll_reload_job(client, job_id, timeout_seconds=10.0)
    assert last_response is not None, "No reload job response"
    assert last_response.get("status") == "success", f"Reload job did not succeed: {last_response}"

    # Verify model_status reports a loaded model (best-effort)
    r_status = client.get("/admin/model_status")
    assert r_status.status_code == 200
    status_json = r_status.json()
    assert status_json.get("loaded") is True

    # Force the app to use our FakeModel for prediction (thread-safe)
    with main._model_lock:
        main._model_instance = FakeModel()

    # Now call /predict with an in-memory JPEG and assert fake prediction
    jpeg_bytes = make_test_jpeg_bytes()
    files = {"file": ("test.jpg", jpeg_bytes, "image/jpeg")}
    r3 = client.post("/predict", files=files)
    assert r3.status_code == 200, r3.text
    j = r3.json()

    # Check structure
    assert "predictions" in j and isinstance(j["predictions"], list)
    assert "metadata" in j and isinstance(j["metadata"], dict)

    # Our FakeModel returns a single prediction entry
    preds = j["predictions"]
    assert len(preds) == 1, f"Expected 1 prediction from FakeModel, got: {preds}"
    assert preds[0]["label"] == "fake"
    assert abs(float(preds[0]["confidence"]) - 1.0) < 1e-6
