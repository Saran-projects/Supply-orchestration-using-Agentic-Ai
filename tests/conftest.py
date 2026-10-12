import os
from pathlib import Path

# Ensure offline / mock execution without real credentials
os.environ.setdefault("OPENAI_API_KEY", "dummy-crewai-mock-key")

import pytest

from sca.config import AppConfig, load_app_config
from sca.db import DuckDBManager, get_duckdb_connection


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def memory_db():
    """Provide an in-memory initialized DuckDB connection."""
    conn = get_duckdb_connection(":memory:")
    yield conn
    conn.close()


@pytest.fixture
def db_manager():
    """Provide a managed in-memory DuckDBManager instance."""
    with DuckDBManager(":memory:") as mgr:
        yield mgr


@pytest.fixture(scope="session")
def app_config(project_root: Path) -> AppConfig:
    """Provide loaded AppConfig from project configs directory."""
    return load_app_config(project_root / "configs")
