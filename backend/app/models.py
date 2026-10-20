"""Tables.

datasets  -> a named, versioned set of labeled examples
examples  -> one input utterance + its expected label (intent + slots)
runs      -> one evaluation of one model on one dataset
results   -> one model prediction for one example inside a run
"""
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def utcnow():
    return datetime.now(timezone.utc)


class Dataset(Base):
    __tablename__ = "datasets"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    examples: Mapped[list["Example"]] = relationship(back_populates="dataset")


class Example(Base):
    __tablename__ = "examples"
    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), index=True)
    text: Mapped[str] = mapped_column(Text)
    # e.g. {"intent": "wrong_item", "order_id": "427", "item": "burger"}
    expected: Mapped[dict] = mapped_column(JSON)
    dataset: Mapped[Dataset] = relationship(back_populates="examples")


class Run(Base):
    __tablename__ = "runs"
    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"))
    model: Mapped[str] = mapped_column(String(100))  # e.g. "llama-3.1-8b-instant"
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending/running/done/failed
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Result(Base):
    __tablename__ = "results"
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"), index=True)
    example_id: Mapped[int] = mapped_column(ForeignKey("examples.id"))
    raw_output: Mapped[str | None] = mapped_column(Text, nullable=True)  # exactly what the model said
    predicted: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # None if output wasn't valid JSON
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
