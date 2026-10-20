"""EvalBoard API. Milestone 1: health + dataset browsing.

You add in later milestones:
  M2: POST /runs            -> start an eval run in the background
  M3: GET  /runs/{id}/metrics -> accuracy, macro-F1, latency, cost
"""
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .db import Base, engine, get_db
from .models import Dataset, Example


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Creates tables if they don't exist. (Later: replace with Alembic migrations.)
    Base.metadata.create_all(engine)
    yield


app = FastAPI(title="EvalBoard", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/datasets")
def list_datasets(db: Session = Depends(get_db)):
    rows = db.execute(
        select(Dataset.id, Dataset.name, Dataset.version, func.count(Example.id))
        .outerjoin(Example)
        .group_by(Dataset.id)
    ).all()
    return [{"id": i, "name": n, "version": v, "num_examples": c} for i, n, v, c in rows]


@app.get("/datasets/{dataset_id}/examples")
def list_examples(dataset_id: int, limit: int = 20, offset: int = 0, db: Session = Depends(get_db)):
    if db.get(Dataset, dataset_id) is None:
        raise HTTPException(status_code=404, detail="dataset not found")
    rows = db.scalars(
        select(Example).where(Example.dataset_id == dataset_id).order_by(Example.id).limit(limit).offset(offset)
    ).all()
    return [{"id": e.id, "text": e.text, "expected": e.expected} for e in rows]
