.PHONY: test lint format app api serve

DATA_ROOT ?= ./data
API_URL ?= http://127.0.0.1:8000

test:
	python -m pytest -q
lint:
	ruff check src scripts tests
format:
	ruff format src scripts tests
app:
	python -m streamlit run app/Home.py

# Local inference API (CPU on a Mac). On GPU hosts also set PRAVAH_REQUIRE_CUDA=1.
api:
	PRAVAH_DATA_ROOT=$(DATA_ROOT) python -m uvicorn --factory pravah.api.main:create_app --host 127.0.0.1 --port 8000

# Dashboard wired to the local API and its predictions folder.
serve:
	PRAVAH_API_URL=$(API_URL) PRAVAH_OUTPUTS_DIR=$(DATA_ROOT)/outputs/predictions \
		python -m streamlit run app/Home.py --server.headless true
