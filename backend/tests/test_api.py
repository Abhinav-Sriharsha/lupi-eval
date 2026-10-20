import os

os.environ["DATABASE_URL"] = "sqlite:///./test_evalboard.db"

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from scripts.seed import main as seed  # noqa: E402


def test_health():
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}


def test_seed_and_list():
    with TestClient(app) as client:
        seed()
        datasets = client.get("/datasets").json()
        assert datasets and datasets[-1]["num_examples"] == 200
        examples = client.get(f"/datasets/{datasets[-1]['id']}/examples?limit=5").json()
        assert len(examples) == 5 and "intent" in examples[0]["expected"]
