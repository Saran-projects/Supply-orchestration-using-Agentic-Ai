# SupplyChain Agents (SCA)

Autonomous Multi-Agent Simulation and Optimization Framework for Global Supply Chains.

SCA integrates discrete-event simulation, classical inventory control baselines (e.g., $(s, S)$, EOQ), LLM-based reasoning agents, and analytical persistence via DuckDB.

---

## Repository Structure

```
supplychain-agents/
├── configs/                  # Central YAML configurations
│   ├── data.yaml             # Raw/processed dataset paths, target, and split settings
│   ├── sim.yaml              # Simulation topology, cost model, and inventory policies
│   ├── experiment.yaml       # Benchmark suite, evaluation metrics, and DB path
│   └── models.yaml           # LLM providers, model allocations, and embeddings
├── data/
│   ├── raw/                  # DataCoSupplyChainDataset.csv, logs, metadata
│   └── processed/            # Cleaned, engineered feature stores
├── src/sca/                  # Core package modules
│   ├── __init__.py           # Package exports and version
│   ├── config.py             # Central Pydantic Settings validated from YAML
│   ├── db.py                 # DuckDB connection helper & 6-table schema
│   ├── data/                 # Ingestion & feature preprocessing
│   ├── sim/                  # Discrete-event supply chain simulation
│   ├── baselines/            # Classical heuristics: (s, S), Base-Stock, EOQ
│   ├── tools/                # Agent tools (order placement, inventory queries)
│   ├── forecast/             # Time-series & demand forecasting
│   ├── llm/                  # LLM clients, prompt templates, structured output
│   ├── agents/               # Role-specialized agents (Inventory, Sourcing, Logistics)
│   ├── systems/              # Coordination systems (Hierarchical, Debate, Graph)
│   ├── eval/                 # KPI computation (OTIF, Bullwhip, Costs, Service Level)
│   ├── logging/              # Telemetry, event streaming, and trajectory tracing
│   └── dashboard/            # Interactive metrics & visualization dashboard
├── tests/                    # Pytest test suite & fixtures
│   ├── conftest.py           # Shared test fixtures (DuckDB, Configs)
│   └── test_smoke.py         # End-to-end smoke tests
├── scripts/                  # Command-line runners and utilities
│   ├── init_db.py            # DuckDB schema initialiser & data verification
│   └── run_sim.py            # Simulation benchmark execution & reporting
├── results/                  # DuckDB database files, run metrics, and plots
├── pyproject.toml            # Package configuration and dependencies
├── Makefile                  # Build, test, lint, and run workflow targets
└── README.md                 # Project documentation
```

---

## Core Components

### 1. Central Configuration Loader (`src/sca/config.py`)
Powered by **Pydantic Settings (v2)**, configuration is loaded and validated against YAML files with environment variable overrides (`SCA_*` prefix):

```python
from sca.config import get_config, load_app_config

# Load unified AppConfig
config = get_config()
print(config.data.dataset_filename)       # 'DataCoSupplyChainDataset.csv'
print(config.sim.cost_model.holding_cost_per_unit_day)  # 0.25
print(config.models.default_model)        # 'gpt-4o-mini'
```

### 2. DuckDB Persistence & Telemetry Schema (`src/sca/db.py`)
Provides analytical storage with automatic table creation and query helpers:

| Table | Purpose |
|---|---|
| `runs` | Experiment-level metadata, configurations, git commit, status |
| `steps` | Simulation time steps, inventory state, actions, rewards |
| `agent_calls` | LLM invocation records: prompt/completion tokens, latency, cost |
| `tool_calls` | Actions executed by agents: input parameters, tool output, errors |
| `messages` | Agent-to-agent debate transcripts and coordination logs |
| `metrics` | Business and operational KPIs (service level, stockout, holding cost) |

### 3. Simulation Scenario Generator (`src/sca/data/scenarios.py`)
Automatically identifies and extracts three realistic regime windows from the processed demand series without data leakage:
- **`stable`**: Window with the lowest coefficient of variation ($CV = \sigma / \mu$).
- **`seasonal`**: Window exhibiting the highest autocorrelation at the dominant period (e.g., weekly lag 7).
- **`shock`**: Window containing the largest demand spike relative to rolling median.

Returns a [`Scenario`](file:///c:/Users/saran/smartsupply/src/sca/data/scenarios.py#L85) object containing:
- Scaled demand array (scaled to configurable `target_mean_demand`, e.g. 100 units).
- Raw unscaled demand array.
- Forecaster training history strictly preceding the window (`train_demand`, `train_dates`) with strict **no-leakage** validation.
- Empirical [`LeadTimeSampler`](file:///c:/Users/saran/smartsupply/src/sca/data/scenarios.py#L43) fitted from DataCo transit statistics.
- Empirical late delivery probability (`0.5483`).

```bash
# Generate scenarios and plots via CLI:
python scripts/generate_scenarios.py
```

---

## Makefile Targets

| Target | Command | Description |
|---|---|---|
| `setup` | `make setup` | Install package and dev dependencies in editable mode |
| `test` | `make test` | Run pytest smoke tests and unit test suite |
| `lint` | `make lint` | Verify code quality and formatting with Ruff |
| `data` | `make data` | Verify raw DataCo dataset and initialize DuckDB schema |
| `baselines` | `make baselines` | Run classical supply chain baseline simulation across 20 seeds |
| `run` | `make run` | Run autonomous multi-agent simulation |
| `report` | `make report` | Generate benchmark comparison report from DuckDB |
| `dashboard` | `make dashboard` | Launch interactive Streamlit monitoring dashboard |
| `reconcile` | `make reconcile` | Reconcile logged LLM token spend across DuckDB databases |

---

## Quickstart

### 1. Environment Setup
```bash
# Using uv (fastest):
uv venv .venv
.venv\Scripts\activate
uv pip install -e ".[dev]" streamlit

# Or using standard pip:
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]" streamlit
```

### 2. Launch Interactive Dashboard (8 Pages, Offline)
```bash
make dashboard
# Or directly with Streamlit:
streamlit run src/sca/dashboard/app.py
```
Open your browser at `http://localhost:8501`. Features:
1. **Overview**: Pipeline topology, 6 systems, metrics taxonomy, base paper.
2. **Real Data & Scenarios**: DataCo Cleats time series, lead times, 90-day scenario windows.
3. **Baselines (Real)**: (S,s) and EOQ evaluated over 20 random seeds (mean ± std).
4. **GLM Real Runs**: S3 and S6 metrics table next to paired baselines (seed 42).
5. **Orchestrator Inspector**: Step-by-step proposals, conflict checks, and orders.
6. **Mock vs GLM**: Plumbing comparison (zero fallbacks, 100% structured output success).
7. **Cost and Plan**: Reconciled spend vs Isoquant console, empirical token ratios.
8. **Agent Conversations**: Chat transcripts, collapsible tool calls, and groundedness verifiers.

### 3. Run Real Experiments & Baselines
```bash
# Run Systems 3 and 6 on GLM-5.3-Flash (shock scenario, seed 42, prompt v2):
python scripts/run_shock_glm_experiment.py

# Run 20-seed classical baselines ((S,s) and EOQ):
make baselines

# Reconcile all-time logged LLM spend against provider console:
make reconcile

# Health check enabled LLM providers:
python -m sca.llm.check
```

### 4. Run Test Suite
```bash
# Run complete test suite (131 tests):
make test

# Run dashboard query tests:
pytest tests/test_dashboard_queries.py -v
```

---

## Models and Keys

The SCA LLM layer supports multiple model families behind a unified, provider-agnostic interface (`LLMClient`), driven entirely through [`configs/models.yaml`](file:///c:/Users/saran/smartsupply/configs/models.yaml):

### Supported Model Families & Adapters
1. **`openai_compatible`**: Uses the OpenAI Python SDK with configurable `base_url` and `api_key_env`.
   - **Isoquant GLM**: `glm-5.3-flash` served via Isoquant (`base_url: https://api.isoquant.ai/v1`, `api_key_env: ISOQUANT_API_KEY`).
   - **OpenAI**: Standard OpenAI endpoints (`base_url: https://api.openai.com/v1`, `api_key_env: OPENAI_API_KEY`).
   - **xAI / Grok**: Grok models via xAI (`base_url: https://api.x.ai/v1`, `api_key_env: XAI_API_KEY`).
2. **`gemini`**: Uses the official Google GenAI SDK (`api_key_env: GOOGLE_API_KEY`).
3. **`mock`**: Deterministic offline test double with canned responses and simulated failures.

### Environment Variables & Security
Keys are read strictly from environment variables via `python-dotenv`:
1. Copy [`.env.example`](file:///c:/Users/saran/smartsupply/.env.example) to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Set your API keys in `.env`:
   ```bash
   ISOQUANT_API_KEY=your_isoquant_key
   OPENAI_API_KEY=your_openai_key
   XAI_API_KEY=your_xai_key
   GOOGLE_API_KEY=your_google_key
   ```
3. **Security Guarantee**: Keys are never stored in YAML files or repository code, `.env` is tracked in [`.gitignore`](file:///c:/Users/saran/smartsupply/.gitignore), and key values are strictly excluded from logging. Missing variables produce an error naming the variable, never any secret data.

### Configuration Guardrails
- **TODO Guardrail**: The loader refuses to execute any enabled model whose `model_id` or prices contain `"TODO_FILL_FROM_PROVIDER_DOCS"` or start with `"TODO"`. Disabled models (`enabled: false`) may hold TODO placeholders safely.
- **Fail-Fast Schema**: Unknown fields in `models.yaml` trigger immediate Pydantic validation errors (`extra="forbid"`).
- **Robustness**: Per-model bounded concurrency, sliding-window requests-per-minute (RPM) rate limits, and exponential backoff with jitter on 429/5xx/timeouts.
- **Structured Output**: Native provider JSON-schema mode when `supports_json_schema: true`, with prompt instruction fallback otherwise. Validation errors trigger a single retry with feedback, raising `StructuredOutputError` on repeated failure.
- **Observability**: Every call logs model name, provider, latency, tokens in/out, cached tokens, cost in USD, retry count, structured output status, and error types to DuckDB `agent_calls`.

### Health Check CLI
Verify connectivity and structured output generation across all enabled models:
```bash
python -m sca.llm.check
```
Exit code is `0` if all enabled models pass, or non-zero (`1`) if any enabled model fails.

For instructions on adding new models without writing code, see [`docs/ADDING_A_MODEL.md`](file:///c:/Users/saran/smartsupply/docs/ADDING_A_MODEL.md).
