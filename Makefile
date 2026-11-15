.PHONY: test lint format app api serve model

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
	PRAVAH_DATA_ROOT=$(DATA_ROOT) python -m uvicorn --factory pravah.api.main:create_app --host 127.0.0.1 --port $(PORT)

# Dashboard wired to the local API and its predictions folder.
serve:
	PRAVAH_API_URL=$(API_URL) PRAVAH_OUTPUTS_DIR=$(DATA_ROOT)/outputs/predictions \
		python -m streamlit run app/Home.py --server.headless true

# Serve YOUR trained model to the dashboard (Live Prediction page).
#   make model                                  # newest checkpoint_best.pt under outputs/models/
#   make model CHECKPOINT=path/to/ckpt.pt INPUT_UNITS=dn DEVICE=mps
# Scenes: put 6- or 13-band Sentinel-2 GeoTIFFs in $(SCENES). Results are written to ./outputs.
CHECKPOINT ?= $(shell ls -t outputs/models/*/checkpoint_best.pt 2>/dev/null | head -1)
SCENES ?= ./data/raw/sentinel2
INPUT_UNITS ?= reflectance
DEVICE ?= auto
PORT ?= 8000
OSM_DIR ?= ./data/processed/osm/thematic
OSM_DATE ?= 2026-01-01
model:
	@test -n "$(CHECKPOINT)" || (echo "No checkpoint: set CHECKPOINT=path/to/checkpoint.pt" && exit 1)
	@! lsof -ti tcp:$(PORT) >/dev/null 2>&1 || (echo "Port $(PORT) is already in use (an older model service?). Stop it with:  lsof -ti tcp:$(PORT) | xargs kill   — or use another port: make model PORT=8001 (then start the app with PRAVAH_API_URL=http://127.0.0.1:8001)" && exit 1)
	@echo "Serving $(CHECKPOINT) · scenes $(SCENES) · input units $(INPUT_UNITS) · device $(DEVICE)"
	PRAVAH_DATA_ROOT=$(DATA_ROOT) PRAVAH_CHECKPOINT=$(CHECKPOINT) PRAVAH_SCENES_DIR=$(SCENES) \
		PRAVAH_OUTPUTS_DIR=./outputs PRAVAH_INPUT_UNITS=$(INPUT_UNITS) PRAVAH_DEVICE=$(DEVICE) \
		PRAVAH_OSM_DIR=$(OSM_DIR) PRAVAH_OSM_DATE=$(OSM_DATE) \
		python -m uvicorn --factory pravah.api.main:create_app --host 127.0.0.1 --port $(PORT)
