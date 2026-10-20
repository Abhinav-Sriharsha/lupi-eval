"""Load data/intents_seed.jsonl into the database as a new dataset.

Run inside Docker:  docker compose exec backend python -m scripts.seed
"""
import json
from pathlib import Path

from app.db import Base, SessionLocal, engine
from app.models import Dataset, Example

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "intents_seed.jsonl"


def main(name: str = "lupi-intents-seed"):
    Base.metadata.create_all(engine)
    rows = [json.loads(line) for line in DATA_FILE.read_text().splitlines() if line.strip()]
    with SessionLocal() as db:
        # New version each time you re-seed, so old runs stay reproducible.
        latest = db.query(Dataset).filter(Dataset.name == name).order_by(Dataset.version.desc()).first()
        ds = Dataset(name=name, version=(latest.version + 1) if latest else 1)
        db.add(ds)
        db.flush()  # assigns ds.id
        db.add_all(Example(dataset_id=ds.id, text=r["text"], expected=r["expected"]) for r in rows)
        db.commit()
        print(f"seeded dataset '{name}' v{ds.version} (id={ds.id}) with {len(rows)} examples")


if __name__ == "__main__":
    main()
