"""
Comprehensive tests for FastAPI ML Service.
Phase 11 — API Endpoints, Validation, Schemas, and Failure Modes.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.config import settings


@pytest.fixture(scope="module")
def client():
    """Test client running with the lifespan context manager."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient):
    """Test GET /health returns ok and model loaded state."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "ml-service"
    assert data["model_loaded"] is True
    assert "LinearSVM" in data["model_name"]
    assert data["model_version"] == "SIC-v1.0"


def test_ready_endpoint(client: TestClient):
    """Test GET /ready returns ready: true when model is loaded."""
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["ready"] is True
    assert data["model_loaded"] is True


def test_model_info_endpoint(client: TestClient):
    """Test GET /model-info returns safe metadata without leaking system paths."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "SarcasmIronyClassifier-LinearSVM"
    assert data["model_version"] == "SIC-v1.0"
    assert "classes" in data
    assert set(data["classes"]) == {"Regular", "Sarcasm", "Irony", "Figurative"}
    assert data["macro_f1"] is not None
    assert data["macro_f1"] > 0.70
    assert "accuracy" in data
    # Ensure no internal filesystem paths are exposed
    json_str = response.text
    assert "C:\\" not in json_str
    assert "d:\\" not in json_str
    assert "/Users/" not in json_str


def test_predict_single_sarcasm(client: TestClient):
    """Test POST /predict with a sarcastic input."""
    payload = {"text": "Oh fantastic, another flat tire on a Monday morning."}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == payload["text"]
    assert data["predicted_class"] in {"Regular", "Sarcasm", "Irony", "Figurative"}
    assert 0.0 <= data["confidence"] <= 1.0
    assert len(data["probabilities"]) == 4
    for cls_name in ["Regular", "Sarcasm", "Irony", "Figurative"]:
        assert cls_name in data["probabilities"]
        assert 0.0 <= data["probabilities"][cls_name] <= 1.0
    # Sum of probabilities should be approximately 1.0
    prob_sum = sum(data["probabilities"].values())
    assert abs(prob_sum - 1.0) < 0.01
    assert data["model_name"] == "SarcasmIronyClassifier-LinearSVM"
    assert data["model_version"] == "SIC-v1.0"
    assert data["latency_ms"] is not None
    assert data["latency_ms"] >= 0.0


def test_predict_single_regular(client: TestClient):
    """Test POST /predict with regular/neutral text."""
    payload = {"text": "The scientific paper was published in the journal yesterday."}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] in {"Regular", "Sarcasm", "Irony", "Figurative"}


def test_predict_validation_empty_string(client: TestClient):
    """Test POST /predict rejects empty string with 422."""
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422
    data = response.json()
    assert "error" in data


def test_predict_validation_whitespace_only(client: TestClient):
    """Test POST /predict rejects whitespace-only string with 422."""
    response = client.post("/predict", json={"text": "   \n\t  "})
    assert response.status_code == 422
    data = response.json()
    assert "error" in data


def test_predict_validation_missing_text(client: TestClient):
    """Test POST /predict rejects missing text key with 422."""
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_validation_invalid_type(client: TestClient):
    """Test POST /predict rejects non-string input with 422."""
    response = client.post("/predict", json={"text": 12345})
    assert response.status_code == 422


def test_predict_validation_exceeding_max_length(client: TestClient):
    """Test POST /predict rejects excessively long text (> 5000 chars) with 422."""
    long_text = "word " * 1200  # > 6000 chars
    response = client.post("/predict", json={"text": long_text})
    assert response.status_code == 422
    data = response.json()
    assert "error" in data


def test_predict_batch_success(client: TestClient):
    """Test POST /predict/batch with multiple texts."""
    payload = {
        "texts": [
            "What a lovely rainy day for a beach picnic.",
            "Please turn off the lights when leaving the room.",
            "Brilliant solution, it only broke everything else.",
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_count"] == 3
    assert len(data["predictions"]) == 3
    assert data["batch_latency_ms"] >= 0.0
    for i, pred in enumerate(data["predictions"]):
        assert pred["text"] == payload["texts"][i]
        assert pred["predicted_class"] in {"Regular", "Sarcasm", "Irony", "Figurative"}
        assert 0.0 <= pred["confidence"] <= 1.0


def test_predict_batch_empty_list(client: TestClient):
    """Test POST /predict/batch rejects empty list with 422."""
    response = client.post("/predict/batch", json={"texts": []})
    assert response.status_code == 422


def test_predict_batch_with_invalid_element(client: TestClient):
    """Test POST /predict/batch rejects batch containing empty element with 422."""
    response = client.post("/predict/batch", json={"texts": ["Valid text", "   "]})
    assert response.status_code == 422


def test_openapi_and_docs_available(client: TestClient):
    """Test OpenAPI schema and Swagger documentation are accessible."""
    openapi_res = client.get("/openapi.json")
    assert openapi_res.status_code == 200
    openapi_data = openapi_res.json()
    assert "paths" in openapi_data
    assert "/health" in openapi_data["paths"]
    assert "/ready" in openapi_data["paths"]
    assert "/model-info" in openapi_data["paths"]
    assert "/predict" in openapi_data["paths"]
    assert "/predict/batch" in openapi_data["paths"]

    docs_res = client.get("/docs")
    assert docs_res.status_code == 200


def test_readiness_failure_when_model_unloaded(client: TestClient):
    """Test GET /ready returns 503 when model is unloaded."""
    original_loaded = app.state.model_loaded
    original_predictor = app.state.predictor
    try:
        app.state.model_loaded = False
        app.state.predictor = None

        ready_res = client.get("/ready")
        assert ready_res.status_code == 503
        data = ready_res.json()
        assert data["ready"] is False
        assert data["model_loaded"] is False

        predict_res = client.post("/predict", json={"text": "Hello world"})
        assert predict_res.status_code == 503

        info_res = client.get("/model-info")
        assert info_res.status_code == 503
    finally:
        # Restore state
        app.state.model_loaded = original_loaded
        app.state.predictor = original_predictor
