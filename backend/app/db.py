"""Database connection: one engine for the app, one session per request."""
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

# Postgres in Docker; falls back to a local SQLite file for quick tests.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./evalboard.db")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False)


class Base(DeclarativeBase):
    pass


def get_db():
    """FastAPI dependency: gives each request its own session and always closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
