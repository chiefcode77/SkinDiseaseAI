# backend/tests/test_admin_reload.py
import sys
import types
import time
from fastapi.testclient import TestClient

# --- create and insert fake model_loader BEFORE importing main ---
fake_mod = types.ModuleType("model_loader")

class FakeModel:
    def __init__(self):
        self.framework = "fake"
        # non-None .model indicates "model_loaded" in the API
        self.model = object()

    def predict_image(self, pil_img):
        # simple deterministic response
        return [{"label": "fake", "confidence": 1.0}]

def get_model():
    return FakeModel()

fake_mod.get_model = get_model

# Insert the fake module so importlib.import_module("model_loader") / reload will find it
sys.modules["model_loader"] = fake_mod

# Now import main (after the fake is in place)
import main  # imports the app and the reload endpoints

def test_reload_job_with_mocked_model_loader():
    """
    Inserted fake 'model_loader' before importing main so the background reload thread
    will find it quickly. Start the non-blocking reload endpoint and poll until finished.
    """
    client = TestClient(main.app)

    # Start reload job (non-blocking)
    r = client.post("/admin/reload_model")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "accepted"
    job_id = data["job_id"]

    # Poll the job until it's finished or we hit a timeout
    status = None
    timeout_seconds = 10.0   # give more time for thread to finish on slower CI
    poll_interval = 0.1
    start = time.time()
    last_response = None

    while time.time() - start < timeout_seconds:
        r2 = client.get(f"/admin/reload_status/{job_id}")
        assert r2.status_code == 200
        last_response = r2.json()
        status = last_response.get("status")
        if status not in ("pending", "running"):
            break
        time.sleep(poll_interval)

    assert status == "success", f"Reload job did not succeed in time: {last_response}"
    assert "result" in last_response and last_response["result"] is not None
    result = last_response["result"]
    # model_framework should indicate our fake loader or at least be present
    assert result.get("model_framework") in ("fake", "dummy", "tensorflow", "unknown")
    # model_loaded should be True because FakeModel.model is non-None
    assert result.get("model_loaded") is True
