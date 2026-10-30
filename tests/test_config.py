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



def test_defaults_allow_local_and_compose_hosts_in_development():
    settings = Settings()
    assert settings.allowed_hosts == ["localhost", "127.0.0.1", "orchestrator"]
    assert settings.environment == "development"


def test_allowed_hosts_and_environment_from_env(monkeypatch):
    monkeypatch.setenv("ALLOWED_HOSTS", "api.example, localhost ,")
    monkeypatch.setenv("ENV", "production")
    settings = Settings.from_env()
    assert settings.allowed_hosts == ["api.example", "localhost"]
    assert settings.environment == "production"


@pytest.mark.parametrize(("var", "value"), [("ENV", "staging"), ("ALLOWED_HOSTS", " , ")])
def test_invalid_environment_or_empty_allowlist_rejected(monkeypatch, var, value):
    monkeypatch.setenv(var, value)
    with pytest.raises(ValidationError):
        Settings.from_env()
