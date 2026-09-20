from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_stats_empty():
    response = client.get("/stats")
    assert response.status_code == 200
    assert "document_count" in response.json()
