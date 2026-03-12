"""
Pytest configuration for Nexus tests.
"""

import os
import pytest
from pathlib import Path

# Load test environment variables
from dotenv import load_dotenv

# Load .env.test if it exists
env_test_path = Path(__file__).parent.parent / ".env.test"
if env_test_path.exists():
    load_dotenv(env_test_path)

# Set testing flag
os.environ["TESTING"] = "True"


@pytest.fixture(autouse=True)
def _unlock_license_gates(monkeypatch):
    """Bypass license gates during tests so gated features are testable."""
    from nexus.licensing import license_gate
    monkeypatch.setattr(license_gate, "check_feature", lambda flag: True)
    monkeypatch.setattr(license_gate, "_license_key", "test-key")


@pytest.fixture(scope="session")
def test_database():
    """Create test database."""
    from nexus.core.database.models import Base
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    # Use test database URL
    database_url = os.getenv("DATABASE_URL", "sqlite:///./test_nexus.db")

    engine = create_engine(database_url)
    Base.metadata.create_all(engine)

    TestingSessionLocal = sessionmaker(bind=engine)

    yield TestingSessionLocal()

    # Cleanup
    Base.metadata.drop_all(engine)
    if database_url.startswith("sqlite"):
        # Remove SQLite test file
        db_file = database_url.replace("sqlite:///", "")
        if os.path.exists(db_file):
            os.remove(db_file)


@pytest.fixture(scope="session")
def redis_client():
    """Create Redis client for testing."""
    import redis

    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    redis_db = int(os.getenv("REDIS_DB", "0"))

    client = redis.Redis(
        host=redis_host,
        port=redis_port,
        db=redis_db,
        decode_responses=True
    )

    # Test connection
    try:
        client.ping()
    except redis.ConnectionError:
        pytest.skip("Redis server not available")

    yield client

    # Cleanup - flush test database
    client.flushdb()


@pytest.fixture(scope="function")
def flask_app():
    """Create Flask app for testing."""
    from nexus.api.api import app

    app.config["TESTING"] = True
    app.config["SECRET_KEY"] = "test-secret-key"

    with app.app_context():
        yield app


@pytest.fixture(scope="function")
def flask_client(flask_app):
    """Create Flask test client."""
    return flask_app.test_client()


@pytest.fixture(scope="function")
def auth_headers():
    """Create authentication headers for testing."""
    # Mock API key for testing
    return {
        "X-API-Key": "test-api-key-12345",
        "Content-Type": "application/json"
    }


@pytest.fixture(scope="function")
def mock_openai_response():
    """Mock OpenAI API response."""
    return {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "This is a test response from the mocked OpenAI API."
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
    }


@pytest.fixture(scope="function")
def mock_anthropic_response():
    """Mock Anthropic API response."""
    return {
        "id": "msg_test123",
        "type": "message",
        "role": "assistant",
        "content": [{
            "type": "text",
            "text": "This is a test response from the mocked Anthropic API."
        }],
        "model": "claude-3-opus-20240229",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 20
        }
    }
