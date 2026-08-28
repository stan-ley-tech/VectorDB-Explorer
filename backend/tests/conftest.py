import os

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

import app.engine_bridge as engine_bridge
import app.models  # noqa: F401  (registers models on Base.metadata)
from app.database import Base

# Exercises the real IVF index build/search path. Only meaningful where the
# vdb_native extension was actually built (CI builds it; a plain local
# `pytest` run on a machine without a C++ toolchain won't have it).
requires_native_engine = pytest.mark.skipif(
    not engine_bridge.NATIVE_ENGINE_AVAILABLE, reason="requires the compiled vdb_native extension"
)

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    os.environ.get("DATABASE_URL", "postgresql+psycopg://vectordb:vectordb@localhost:5432/vectordb_test"),
)


def _postgres_available() -> bool:
    try:
        # A short connect_timeout matters here: with no Postgres listening,
        # a plain refused connection fails instantly, but some environments
        # (e.g. a filtered port) leave the default TCP connect to hang for
        # minutes, which would turn "just run pytest locally" into a stall.
        engine = create_engine(TEST_DATABASE_URL, connect_args={"connect_timeout": 2})
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
        return True
    except Exception:
        return False


# API/DB integration tests need a real Postgres with pgvector (JSONB
# containment, the vector column type and its distance operators have no
# sqlite equivalent worth faking), so they're skipped rather than failed
# when nothing is reachable. CI runs them for real against a postgres
# service container; see .github/workflows/ci.yml.
requires_postgres = pytest.mark.skipif(
    not _postgres_available(), reason="requires a reachable Postgres+pgvector instance"
)


@pytest.fixture(scope="session")
def db_engine():
    engine = create_engine(TEST_DATABASE_URL)
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture()
def db_session(db_engine):
    session_factory = sessionmaker(bind=db_engine)
    session = session_factory()
    # Clean slate per test. Reverse sorted-table order so deletes respect
    # foreign keys without needing per-table CASCADE logic here.
    for table in reversed(Base.metadata.sorted_tables):
        session.execute(table.delete())
    session.commit()
    yield session
    session.close()


@pytest.fixture()
def client(db_session):
    from fastapi.testclient import TestClient

    from app.database import get_db
    from app.main import app

    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
