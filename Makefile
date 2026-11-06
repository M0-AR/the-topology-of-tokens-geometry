PYTHON ?= .venv/bin/python
TOPOBPE ?= .venv/bin/topobpe
CONFIG ?= configs/default.yaml

.PHONY: install test lint smoke download tokenizers features lm market-data market-features market-lm analysis figures manifest all docker-config

install:
	uv venv --python 3.12 --system-site-packages .venv
	uv pip install --python $(PYTHON) -r requirements.lock
	uv pip install --python $(PYTHON) --no-deps -e .

test:
	$(PYTHON) -m pytest -q

lint:
	.venv/bin/ruff check src tests scripts

smoke:
	$(TOPOBPE) all --config configs/smoke.yaml --skip-market

download:
	$(TOPOBPE) download --config $(CONFIG)

tokenizers:
	$(TOPOBPE) tokenizers --config $(CONFIG)

features:
	$(TOPOBPE) features --config $(CONFIG)

lm:
	$(TOPOBPE) lm --config $(CONFIG)

market-data:
	$(TOPOBPE) market-data --config $(CONFIG)

market-features:
	$(TOPOBPE) market-features --config $(CONFIG)

market-lm:
	$(TOPOBPE) market-lm --config $(CONFIG)

analysis:
	$(TOPOBPE) analysis --config $(CONFIG)

figures:
	$(TOPOBPE) figures --config $(CONFIG)

manifest:
	$(TOPOBPE) manifest --config $(CONFIG)

all:
	$(TOPOBPE) all --config $(CONFIG)

docker-config:
	docker compose config --quiet
