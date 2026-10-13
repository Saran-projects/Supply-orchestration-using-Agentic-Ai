.PHONY: help setup test lint data baselines run report dashboard reconcile clean

PYTHON ?= python
PIP ?= pip

help:
	@echo "SupplyChain Agents (SCA) Makefile commands:"
	@echo "  make setup     - Install dependencies and local editable package"
	@echo "  make test      - Run test suite with pytest"
	@echo "  make lint      - Check code style and formatting with ruff"
	@echo "  make data      - Initialize DuckDB schema and verify DataCo raw dataset"
	@echo "  make baselines - Run classical supply chain baseline simulation"
	@echo "  make run       - Run autonomous multi-agent simulation"
	@echo "  make report    - Generate benchmark comparison report from DuckDB"
	@echo "  make dashboard - Launch interactive monitoring dashboard"
	@echo "  make reconcile - Reconcile logged LLM token spend across DuckDB databases"

setup:
	$(PIP) install -e ".[dev]"

test:
	$(PYTHON) -m pytest tests -v

lint:
	$(PYTHON) -m ruff check src tests configs scripts

data:
	$(PYTHON) scripts/process_data.py

baselines:
	$(PYTHON) -m sca.eval.runner --config configs/experiment.yaml

run:
	$(PYTHON) scripts/run_sim.py --mode agentic

report:
	$(PYTHON) scripts/run_sim.py --mode report

dashboard:
	$(PYTHON) scripts/run_sim.py --mode dashboard

reconcile:
	$(PYTHON) -m sca.llm.reconcile --recompute

clean:
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]; [p.unlink() for p in pathlib.Path('.').rglob('*.pyc')]"

