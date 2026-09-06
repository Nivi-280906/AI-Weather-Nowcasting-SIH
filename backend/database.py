"""
Database configuration.
Uses SQLite for zero-setup local/demo running. Swap DATABASE_URL for
Postgres/PostGIS in production (schema is written to be portable).
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = "sqlite:///./nowcast.db"

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    import models  # noqa: F401 (ensures models are registered on Base)
    Base.metadata.create_all(bind=engine)
