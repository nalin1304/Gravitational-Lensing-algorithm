"""Tests for the alternative non-Streamlit web UI served by FastAPI."""

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_ui_index_served() -> None:
    """`/ui` should serve the static HTML frontend."""
    response = client.get("/ui")
    assert response.status_code == 200
    assert "Lensing Research Workbench" in response.text


def test_ui_static_css_served() -> None:
    """Stylesheet should be publicly accessible."""
    response = client.get("/ui-static/styles.css")
    assert response.status_code == 200
    assert "--bg" in response.text


def test_ui_static_js_served() -> None:
    """Frontend JavaScript should be publicly accessible."""
    response = client.get("/ui-static/app.js")
    assert response.status_code == 200
    assert "lensing_workbench_runs" in response.text
