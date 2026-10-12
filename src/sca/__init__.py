"""Supply Chain Agents (SCA).

Autonomous multi-agent simulation and optimization framework for supply chains.
"""

from sca.config import (
    AppConfig,
    DataConfig,
    ExperimentConfig,
    ModelEntryConfig,
    ModelsConfig,
    SimConfig,
    get_config,
    load_app_config,
    load_data_config,
    load_experiment_config,
    load_models_config,
    load_sim_config,
)
from sca.db import (
    DuckDBManager,
    get_duckdb_connection,
    init_schema,
    verify_schema,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "AppConfig",
    "DataConfig",
    "SimConfig",
    "ExperimentConfig",
    "ModelEntryConfig",
    "ModelsConfig",
    "get_config",
    "load_app_config",
    "load_data_config",
    "load_sim_config",
    "load_experiment_config",
    "load_models_config",
    "get_duckdb_connection",
    "init_schema",
    "verify_schema",
    "DuckDBManager",
]
