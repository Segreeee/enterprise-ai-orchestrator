from pathlib import Path

import pytest
from pydantic import ValidationError

from orchestrator.config import Settings


def test_defaults():
    settings = Settings()
    assert str(settings.ollama_host) == "http://localhost:11434/"
    assert settings.chroma_path == Path("chroma_db")
    assert settings.model_name == "llama3.2"


def test_from_env_overrides(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://ollama:11434")
    monkeypatch.setenv("CHROMA_PATH", "/data/chroma")
    monkeypatch.setenv("MODEL_NAME", "mistral")
    settings = Settings.from_env()
    assert settings.ollama_host.host == "ollama"
    assert settings.chroma_path == Path("/data/chroma")
    assert settings.model_name == "mistral"


def test_from_env_keeps_defaults_when_unset(monkeypatch):
    for var in ("OLLAMA_HOST", "CHROMA_PATH", "MODEL_NAME"):
        monkeypatch.delenv(var, raising=False)
    assert Settings.from_env() == Settings()


@pytest.mark.parametrize("bad_url", ["not-a-url", "ftp://ollama:11434", ""])
def test_invalid_ollama_host_rejected(bad_url):
    with pytest.raises(ValidationError):
        Settings(ollama_host=bad_url)


def test_settings_are_immutable_and_strict():
    with pytest.raises(ValidationError):
        Settings(unknown_field="x")
    settings = Settings()
    with pytest.raises(ValidationError):
        settings.model_name = "other"
