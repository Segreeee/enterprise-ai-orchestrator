PYTHON ?= python
IMAGE  ?= enterprise-ai-orchestrator

.PHONY: venv install test run docker-build docker-run clean

venv:
	$(PYTHON) -m venv .venv

install:
	$(PYTHON) -m pip install -r requirements.txt -r requirements-dev.txt

test:
	$(PYTHON) -m pytest -q

run:
	$(PYTHON) -m uvicorn orchestrator.api.app:app --app-dir src --reload

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -p 8000:8000 $(IMAGE)

clean:
	$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p) for p in pathlib.Path(\".\").rglob(\"*\") if p.name in (\"__pycache__\", \".pytest_cache\")]"
