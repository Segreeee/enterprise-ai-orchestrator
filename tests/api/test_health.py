import io
import json

from fastapi.testclient import TestClient
from loguru import logger

from orchestrator import __version__
from orchestrator.api.app import configure_logging, create_app


def test_health_returns_ok_and_version():
    response = TestClient(create_app()).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}


def test_configure_logging_writes_one_json_object_per_line():
    buffer = io.StringIO()
    configure_logging(buffer)
    try:
        logger.debug("below the default level")
        with logger.contextualize(conversation_id="c1"):
            logger.info("conversation finished", turns=2)
        lines = buffer.getvalue().splitlines()
    finally:
        configure_logging()

    assert len(lines) == 1
    record = json.loads(lines[0])["record"]
    assert record["message"] == "conversation finished"
    assert record["level"]["name"] == "INFO"
    assert record["extra"] == {"conversation_id": "c1", "turns": 2}
