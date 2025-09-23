# tests/test_api.py
import pytest
from httpx import AsyncClient
from app.main import app

@pytest.mark.anyio
async def test_predict_success():
    async with AsyncClient(app=app, base_url="http://test") as client:
        resp = await client.post("/predict", json={
        "features": [0.5, 1, 1.5]
    })
    assert resp.status_code == 200
    assert {"predictions": [1, 2, 3]} == resp.json()

@pytest.mark.anyio
async def test_predict_unprocessable_entity():
    async with AsyncClient(app=app, base_url="http://test") as client:
        resp = await client.post("/predict", json={
        "feature1": 3.5,
        "feature2": 1.2,
        "feature3": 4
    })
    assert resp.status_code == 422

@pytest.mark.anyio
async def test_predict_failure():
    async with AsyncClient(app=app, base_url="http://test") as client:
        resp = await client.post("/predict", json={
        "features": [0.5, 1, 1.5]
    })
    assert resp.status_code == 200
    assert {"predictions": [1, 4, 3]} == resp.json()