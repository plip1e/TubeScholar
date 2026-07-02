"""First test: prove the app boots and the health endpoint responds.

Run with:  pytest   (from the project root)
"""

from fastapi.testclient import TestClient

from tube_scholar.main import app

client = TestClient(app)


def test_health_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_routes_registered():
    # Prove the real endpoints are wired without invoking the LLM/graph (which needs
    # API keys). We just check the route table.
    paths = {route.path for route in app.routes}
    assert "/chat" in paths
    assert "/ingest" in paths