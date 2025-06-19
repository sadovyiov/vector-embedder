import pytest
from fastapi.testclient import TestClient
from main import app
import os

client = TestClient(app)

# Підстав API-ключ із середовища або заміни вручну
API_KEY = os.getenv("KEY", "test-key")

def test_healthz():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_embed_unauthorized():
    response = client.post("/embed", json={"text": "hello"})
    assert response.status_code == 401

def test_embed_authorized():
    response = client.post(
        "/embed",
        headers={"Key": API_KEY},
        json={"text": "Shimano Twin Power"}
    )
    assert response.status_code == 200
    assert "embedding" in response.json()
    assert isinstance(response.json()["embedding"], list)