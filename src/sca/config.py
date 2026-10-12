"""Central configuration loader for Supply Chain Agents (SCA).

Provides Pydantic Settings models validated against YAML configuration files:
- data.yaml -> DataConfig
- sim.yaml -> SimConfig
- experiment.yaml -> ExperimentConfig
- models.yaml -> ModelsConfig
- AppConfig -> Unified composition
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load environment variables once at module import
load_dotenv(find_dotenv(usecwd=True))


def find_project_root() -> Path:
    """Find the project root by looking for pyproject.toml or configs directory."""
    current = Path.cwd().resolve()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists() or (parent / "configs").exists():
            return parent
    return Path(__file__).resolve().parent.parent.parent


def load_yaml_file(file_path: Path | str) -> Dict[str, Any]:
    """Safely load and parse a YAML file."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


# ---------------------------------------------------------------------------
# Data Configuration Models
# ---------------------------------------------------------------------------

class DataConfig(BaseModel):
    """Configuration for data loading, paths, features, and preprocessing."""

    raw_data_dir: Path = Field(default=Path("data/raw"), description="Raw data directory")
    processed_data_dir: Path = Field(default=Path("data/processed"), description="Processed data directory")
    dataset_filename: str = Field(default="DataCoSupplyChainDataset.csv", description="Main dataset file name")
    access_logs_filename: str = Field(default="tokenized_access_logs.csv", description="Access logs file name")
    description_filename: str = Field(default="DescriptionDataCoSupplyChain.csv", description="Dataset description file name")
    encoding: str = Field(default="latin1", description="CSV file encoding")

    target_column: str = Field(default="Late_delivery_risk", description="Target classification/prediction column")
    date_columns: List[str] = Field(default_factory=list, description="Columns representing timestamps")
    categorical_columns: List[str] = Field(default_factory=list, description="Categorical feature columns")
    numeric_columns: List[str] = Field(default_factory=list, description="Numeric feature columns")

    train_split: float = Field(default=0.8, ge=0.0, le=1.0)
    val_split: float = Field(default=0.1, ge=0.0, le=1.0)
    test_split: float = Field(default=0.1, ge=0.0, le=1.0)
    random_seed: int = Field(default=42)
    handle_missing_values: str = Field(default="impute")
    drop_high_cardinality: bool = Field(default=True)
    max_cardinality_levels: int = Field(default=50)

    # Series Selection & Aggregation Settings
    granularity: str = Field(default="daily", description="Time granularity: 'daily' or 'weekly'")
    category: Optional[str] = Field(default=None, description="Selected category for simulation/forecasting series")
    market: Optional[str] = Field(default=None, description="Selected market for simulation/forecasting series")
    rolling_window: int = Field(default=7, ge=1, description="Rolling window size for trend smoothing")
    eda_dir: Path = Field(default=Path("results/eda"), description="Directory to save EDA plots and summary")
    processed_daily_filename: str = Field(default="demand_daily.parquet", description="Processed daily demand parquet file")
    processed_weekly_filename: str = Field(default="demand_weekly.parquet", description="Processed weekly demand parquet file")

    # Scenario Generation Settings
    scenario_window_length: int = Field(default=40, gt=0, description="Window length in steps for scenario simulation")
    min_train_history: int = Field(default=90, ge=0, description="Minimum training steps required before the scenario window")
    target_mean_demand: float = Field(default=100.0, gt=0.0, description="Target mean demand to scale simulator units")
    dominant_period: int = Field(default=7, gt=0, description="Dominant seasonality lag period (e.g. 7 for weekly in daily)")
    scenarios_output_dir: Path = Field(default=Path("results/scenarios"), description="Directory to save scenario plots and data")

    # Scenario Generation v2 Settings
    shock_level_shift_enabled: bool = Field(default=True, description="Enable injected level shift for shock scenario")
    shock_level_shift_multiplier: float = Field(default=1.5, ge=1.0, description="Multiplier for injected level shift")
    shock_level_shift_duration: int = Field(default=10, gt=0, description="Duration in steps of injected level shift")
    shock_level_shift_start: int = Field(default=15, ge=0, description="Start step for injected level shift")
    shock_provenance_label: str = Field(default="dataco_window_plus_injected_level_shift_v2", description="Provenance metadata label")

    @field_validator("raw_data_dir", "processed_data_dir", "eda_dir", "scenarios_output_dir", mode="before")
    @classmethod
    def _coerce_path(cls, v: Any) -> Path:
        return Path(v) if not isinstance(v, Path) else v

    @property
    def raw_dataset_path(self) -> Path:
        """Absolute or resolved path to raw dataset CSV."""
        return self.raw_data_dir / self.dataset_filename


# ---------------------------------------------------------------------------
# Simulation Configuration Models
# ---------------------------------------------------------------------------

class SupplierNode(BaseModel):
    id: str
    name: str
    lead_time_days: int = Field(gt=0)
    lead_time_std: float = Field(ge=0.0)
    reliability: float = Field(ge=0.0, le=1.0)


class DistributionCenterNode(BaseModel):
    id: str
    name: str
    capacity_units: int = Field(gt=0)
    initial_inventory: int = Field(ge=0)


class TopologyNodes(BaseModel):
    suppliers: List[SupplierNode] = Field(default_factory=list)
    distribution_centers: List[DistributionCenterNode] = Field(default_factory=list)


class CostModelConfig(BaseModel):
    holding_cost_per_unit_day: float = Field(default=0.25, ge=0.0)
    stockout_penalty_per_unit: float = Field(default=15.00, ge=0.0)
    order_placement_fixed_cost: float = Field(default=100.00, ge=0.0)
    expedited_shipping_multiplier: float = Field(default=2.50, ge=1.0)
    carbon_cost_per_ton_km: float = Field(default=0.05, ge=0.0)


class InventoryPolicyConfig(BaseModel):
    policy_type: str = Field(default="s_S")
    reorder_point: int = Field(default=2000, ge=0)
    order_up_to: int = Field(default=8000, ge=0)
    service_level_target: float = Field(default=0.95, ge=0.0, le=1.0)
    safety_stock_factor: float = Field(default=1.65, ge=0.0)


class FourEchelonConfig(BaseModel):
    """Configuration for 4-echelon serial beer-game simulation."""

    initial_inventory: float = Field(default=200.0, ge=0.0)
    initial_backlog: float = Field(default=0.0, ge=0.0)
    holding_cost_per_unit: float = Field(default=0.50, ge=0.0)
    backlog_cost_multiplier: float = Field(default=10.0, ge=1.0, description="Backlog cost multiple relative to holding cost")
    backlog_cost_per_unit: float = Field(default=5.00, ge=0.0)
    manufacturer_capacity: float = Field(default=300.0, gt=0.0)
    history_window: int = Field(default=14, gt=0)
    default_lead_time: int = Field(default=2, ge=0)
    late_delivery_delay: int = Field(default=1, ge=0)
    horizon_steps: int = Field(default=40, gt=0, description="Common simulation horizon steps")
    warmup_steps: int = Field(default=5, ge=0, description="Warm-up steps discarded during metric evaluation")
    steady_state_init: bool = Field(default=False, description="Initialize pipelines and states at steady state")
    alpha: float = Field(default=0.8, ge=0.0, le=1.0, description="Demand forecast smoothing factor alpha")
    beta: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Order smoothing parameter beta (defaults to alpha if None)")
    safety_factor_z: float = Field(default=1.65, ge=0.0, description="Safety stock z factor")
    include_lead_time_variance: bool = Field(default=True, description="Whether to include lead-time variance in safety stock")
    rate_limit_delta: float = Field(default=0.5, ge=0.0, description="Rate limit delta for order changes in validate_order")
    forecast_cap_multiple: float = Field(default=2.0, ge=1.0, description="Forecast-relative order cap multiple in validate_order")
    enable_smoothed_target_conflict: bool = Field(default=False, description="Flag for orchestrator smoothed target conflict")
    smoothed_target_threshold: float = Field(default=0.25, ge=0.0, description="Deviation threshold from smoothed target")
    independent_demand_paths: bool = Field(default=True, description="Generate independent demand paths per seed via circular block bootstrap")
    tuned_eoq_q_mult: float = Field(default=0.75, ge=0.0, description="Tuned EOQ order quantity multiplier")
    tuned_eoq_rop_factor: float = Field(default=1.2, ge=0.0, description="Tuned EOQ reorder point factor")
    tuned_ss_s_factor: float = Field(default=1.2, ge=0.0, description="Tuned (s,S) reorder point factor s")
    tuned_ss_S_factor: float = Field(default=0.25, ge=0.0, description="Tuned (s,S) order-up-to factor S")
    tuning_seeds_used: List[int] = Field(default_factory=lambda: [100, 101, 102, 103, 104])


class SimConfig(BaseModel):
    """Simulation environment settings and business rules."""

    simulation_name: str = Field(default="dataco_inventory_sim")
    time_horizon_days: int = Field(default=40, gt=0)
    step_size_days: int = Field(default=1, gt=0)
    warmup_days: int = Field(default=5, ge=0)
    warmup_steps: int = Field(default=5, ge=0)
    random_seed: int = Field(default=42)
    independent_demand_paths: bool = Field(default=True, description="Generate independent demand paths per seed")

    nodes: TopologyNodes = Field(default_factory=TopologyNodes)
    cost_model: CostModelConfig = Field(default_factory=CostModelConfig)
    default_policy: InventoryPolicyConfig = Field(default_factory=InventoryPolicyConfig)
    four_echelon: FourEchelonConfig = Field(default_factory=FourEchelonConfig)

    stochastic_demand: bool = Field(default=True)
    disruption_probability_per_step: float = Field(default=0.02, ge=0.0, le=1.0)
    disruption_duration_days_max: int = Field(default=4, ge=0)
    late_delivery_penalty_rate: float = Field(default=0.10, ge=0.0)


# ---------------------------------------------------------------------------
# Experiment Configuration Models
# ---------------------------------------------------------------------------

class SystemSpec(BaseModel):
    id: str
    type: str
    enabled: bool = True


class ConstraintsConfig(BaseModel):
    min_service_level: float = Field(default=0.90, ge=0.0, le=1.0)
    max_allowable_stockout_units: int = Field(default=500, ge=0)
    budget_limit_usd: float = Field(default=500000.00, ge=0.0)


class ExperimentConfig(BaseModel):
    """Benchmark experiment tracking, persistence, and evaluation targets."""

    experiment_name: str = Field(default="sca_benchmark_experiment")
    description: str = Field(default="Benchmark multi-agent supply chain orchestration")
    version: str = Field(default="0.1.0")
    tags: List[str] = Field(default_factory=list)

    db_path: Path = Field(default=Path("results/sca_metrics.duckdb"))
    output_dir: Path = Field(default=Path("results"))
    save_step_artifacts: bool = Field(default=True)
    save_conversation_logs: bool = Field(default=True)
    log_level: str = Field(default="INFO")
    seed: int = Field(default=42)

    systems: List[SystemSpec] = Field(default_factory=list)
    scenarios: List[str] = Field(default_factory=lambda: ["stable", "seasonal", "shock"])
    seeds: Union[int, List[int]] = Field(default=20)
    concurrency: Dict[str, int] = Field(
        default_factory=lambda: {"max_workers_non_llm": 8, "max_workers_llm": 2}
    )
    metrics: List[str] = Field(default_factory=list)
    constraints: ConstraintsConfig = Field(default_factory=ConstraintsConfig)

    @field_validator("db_path", "output_dir", mode="before")
    @classmethod
    def _coerce_path(cls, v: Any) -> Path:
        return Path(v) if not isinstance(v, Path) else v


# ---------------------------------------------------------------------------
# Models Configuration Models (LLM & Embeddings)
# ---------------------------------------------------------------------------

class ProviderConfig(BaseModel):
    api_key_env_var: Optional[str] = None
    base_url: Optional[str] = None
    default_model: str
    fallback_model: Optional[str] = None
    timeout_seconds: int = Field(default=60, gt=0)
    max_retries: int = Field(default=3, ge=0)


class AgentModelAllocation(BaseModel):
    provider: str = Field(default="openai")
    model: str = Field(default="gpt-4o-mini")
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, gt=0)


class EmbeddingsConfig(BaseModel):
    provider: str = Field(default="openai")
    model: str = Field(default="text-embedding-3-small")
    dimension: int = Field(default=1536, gt=0)


class ModelEntryConfig(BaseModel):
    """Pydantic model representing an individual model specification."""

    model_config = ConfigDict(extra="forbid")

    name: str
    provider: str
    model_id: str
    base_url: Optional[str] = None
    api_key_env: str
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, gt=0)
    timeout_s: float = Field(default=60.0, gt=0.0)
    price_per_1k_input: Union[float, str]
    price_per_1k_output: Union[float, str]
    supports_json_schema: bool = True
    enabled: bool = True
    max_concurrency: int = Field(default=10, gt=0)
    requests_per_minute: int = Field(default=60, gt=0)

    @model_validator(mode="after")
    def validate_todo_guard(self) -> "ModelEntryConfig":
        if self.enabled:
            # Check model_id
            if isinstance(self.model_id, str) and (
                self.model_id == "TODO_FILL_FROM_PROVIDER_DOCS" or self.model_id.strip().startswith("TODO")
            ):
                raise ValueError(
                    f"Model '{self.name}' is enabled but model_id is still a placeholder: '{self.model_id}'. "
                    "Update model_id or set enabled: false."
                )
            # Check price_per_1k_input
            if isinstance(self.price_per_1k_input, str):
                if "TODO" in self.price_per_1k_input:
                    raise ValueError(
                        f"Model '{self.name}' is enabled but price_per_1k_input is still a placeholder: "
                        f"'{self.price_per_1k_input}'. Update price or set enabled: false."
                    )
                try:
                    self.price_per_1k_input = float(self.price_per_1k_input)
                except ValueError as err:
                    raise ValueError(
                        f"Model '{self.name}' has invalid price_per_1k_input: '{self.price_per_1k_input}'"
                    ) from err
            # Check price_per_1k_output
            if isinstance(self.price_per_1k_output, str):
                if "TODO" in self.price_per_1k_output:
                    raise ValueError(
                        f"Model '{self.name}' is enabled but price_per_1k_output is still a placeholder: "
                        f"'{self.price_per_1k_output}'. Update price or set enabled: false."
                    )
                try:
                    self.price_per_1k_output = float(self.price_per_1k_output)
                except ValueError as err:
                    raise ValueError(
                        f"Model '{self.name}' has invalid price_per_1k_output: '{self.price_per_1k_output}'"
                    ) from err
        return self


class ModelsConfig(BaseModel):
    """LLM provider, models registry, and agent allocations."""

    model_config = ConfigDict(extra="forbid")

    default_provider: str = Field(default="openai_compatible")
    default_model: str = Field(default="glm-isoquant")
    mock_mode: bool = Field(default=False)
    opik_enabled: bool = Field(default=False, description="Enable Opik observability tracing")
    cache_dir: Path = Field(default=Path(".cache/llm"), description="Local disk cache directory")
    max_retries: int = Field(default=3, ge=0)
    models: List[ModelEntryConfig] = Field(default_factory=list)
    providers: Dict[str, ProviderConfig] = Field(default_factory=dict)
    agents: Dict[str, AgentModelAllocation] = Field(default_factory=dict)
    embeddings: Optional[EmbeddingsConfig] = None

    @field_validator("cache_dir", mode="before")
    @classmethod
    def _coerce_path(cls, v: Any) -> Path:
        return Path(v) if not isinstance(v, Path) else v

    def get_model(self, name: str) -> Optional[ModelEntryConfig]:
        """Look up a model entry by name."""
        for m in self.models:
            if m.name == name:
                return m
        return None


# ---------------------------------------------------------------------------
# Central AppConfig (Pydantic Settings)
# ---------------------------------------------------------------------------

class AppConfig(BaseSettings):
    """Unified application settings combining all YAML configs."""

    model_config = SettingsConfigDict(
        env_prefix="SCA_",
        case_sensitive=False,
        arbitrary_types_allowed=True,
    )

    env: str = Field(default="development")
    data: DataConfig
    sim: SimConfig
    experiment: ExperimentConfig
    models: ModelsConfig


# ---------------------------------------------------------------------------
# Loader Functions
# ---------------------------------------------------------------------------

def resolve_configs_dir(configs_dir: Optional[Path | str] = None) -> Path:
    """Resolve configs directory path."""
    if configs_dir:
        resolved = Path(configs_dir)
        if resolved.exists():
            return resolved
    root = find_project_root()
    return root / "configs"


def load_data_config(configs_dir: Optional[Path | str] = None) -> DataConfig:
    """Load and validate DataConfig from data.yaml."""
    c_dir = resolve_configs_dir(configs_dir)
    data = load_yaml_file(c_dir / "data.yaml")
    return DataConfig.model_validate(data)


def load_sim_config(configs_dir: Optional[Path | str] = None) -> SimConfig:
    """Load and validate SimConfig from sim.yaml."""
    c_dir = resolve_configs_dir(configs_dir)
    data = load_yaml_file(c_dir / "sim.yaml")
    return SimConfig.model_validate(data)


def load_experiment_config(configs_dir: Optional[Path | str] = None) -> ExperimentConfig:
    """Load and validate ExperimentConfig from experiment.yaml."""
    c_dir = resolve_configs_dir(configs_dir)
    data = load_yaml_file(c_dir / "experiment.yaml")
    return ExperimentConfig.model_validate(data)


def load_models_config(configs_dir: Optional[Path | str] = None) -> ModelsConfig:
    """Load and validate ModelsConfig from models.yaml."""
    c_dir = resolve_configs_dir(configs_dir)
    data = load_yaml_file(c_dir / "models.yaml")
    return ModelsConfig.model_validate(data)


@lru_cache(maxsize=1)
def load_app_config(configs_dir: Optional[Path | str] = None) -> AppConfig:
    """Load and validate all configurations into unified AppConfig."""
    c_dir = resolve_configs_dir(configs_dir)
    return AppConfig(
        data=load_data_config(c_dir),
        sim=load_sim_config(c_dir),
        experiment=load_experiment_config(c_dir),
        models=load_models_config(c_dir),
    )


def get_config() -> AppConfig:
    """Get the cached AppConfig instance."""
    return load_app_config()
