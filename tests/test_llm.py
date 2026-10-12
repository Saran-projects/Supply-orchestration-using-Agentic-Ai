"""Unit tests for LLMClient and MockLLM test double (offline testing only).

Tests:
- models.yaml validation (including TODO guard and fail-fast on unknown fields)
- Missing environment variable error messages (naming variable, never secret)
- Disk cache hit/miss semantics and cost=0 logging
- Transient 429 retry with exponential backoff and jitter
- Structured-output single retry on validation error and StructuredOutputError
- Cost calculation from entry prices and fallback pricing table
- Security: ensuring API key values never appear in logs
- Tokenizer fallback when token counts are missing (usage_estimated=True)
"""

from __future__ import annotations

import logging

import duckdb
import pytest
from pydantic import BaseModel, Field, ValidationError

from sca.config import (
    ModelEntryConfig,
    ModelsConfig,
    ProviderConfig,
    load_models_config,
)
from sca.llm.client import (
    LLMClient,
    LLMResponse,
    StructuredOutputError,
    calculate_cost,
)


class SampleDecisionModel(BaseModel):
    """Sample structured output model for testing."""

    recommended_order: float = Field(..., description="Recommended order units")
    rationale: str = Field(..., description="Explanation of replenishment reasoning")
    risk_level: str = Field(default="low", description="Assessed supply disruption risk")


@pytest.fixture
def temp_cache(tmp_path):
    """Temporary disk cache directory."""
    return tmp_path / "llm_cache"


@pytest.fixture
def temp_db(tmp_path):
    """Temporary DuckDB path for testing agent_calls."""
    return tmp_path / "test_llm_metrics.duckdb"


@pytest.fixture
def mock_client(temp_cache, temp_db):
    """LLMClient configured in mock mode for offline testing."""
    cfg = ModelsConfig(
        default_provider="mock",
        default_model="mock",
        mock_mode=True,
        opik_enabled=False,
        cache_dir=temp_cache,
        models=[
            ModelEntryConfig(
                name="mock",
                provider="mock",
                model_id="mock-model",
                api_key_env="MOCK_API_KEY",
                temperature=0.0,
                max_tokens=2048,
                timeout_s=10.0,
                price_per_1k_input=0.0,
                price_per_1k_output=0.0,
                supports_json_schema=True,
                enabled=True,
                max_concurrency=10,
                requests_per_minute=60,
            ),
            ModelEntryConfig(
                name="glm-isoquant",
                provider="openai_compatible",
                model_id="glm-5.3-flash",
                base_url="https://api.isoquant.ai/v1",
                api_key_env="ISOQUANT_API_KEY",
                temperature=0.2,
                max_tokens=2048,
                timeout_s=60.0,
                price_per_1k_input=0.000154,
                price_per_1k_output=0.00044,
                supports_json_schema=True,
                enabled=True,
                max_concurrency=10,
                requests_per_minute=60,
            ),
        ],
        providers={
            "mock": ProviderConfig(default_model="mock-model"),
            "openai": ProviderConfig(default_model="gpt-4o-mini"),
        },
    )
    return LLMClient(
        config=cfg,
        mock_mode=True,
        cache_dir=temp_cache,
        db_path=temp_db,
    )


# ---------------------------------------------------------------------------
# 1. models.yaml Validation & TODO Guard Tests
# ---------------------------------------------------------------------------
def test_models_yaml_loading():
    """Verify that configs/models.yaml loads with four model families and mock entry."""
    cfg = load_models_config()
    assert isinstance(cfg, ModelsConfig)
    model_names = [m.name for m in cfg.models]
    assert "glm-isoquant" in model_names
    assert "openai-main" in model_names
    assert "grok-main" in model_names
    assert "gemini-main" in model_names
    assert "mock" in model_names

    glm = cfg.get_model("glm-isoquant")
    assert glm is not None
    assert glm.enabled is True
    assert glm.provider == "openai_compatible"
    assert glm.base_url == "https://api.isoquant.ai/v1"
    assert glm.price_per_1k_input == 0.000154
    assert glm.price_per_1k_output == 0.00044

    # Disabled models with TODO must be permitted
    openai_main = cfg.get_model("openai-main")
    assert openai_main is not None
    assert openai_main.enabled is False
    assert openai_main.model_id == "TODO_FILL_FROM_PROVIDER_DOCS"


def test_todo_guard_rejects_enabled_model():
    """Verify that the loader refuses to run any enabled model whose model_id or prices are TODO."""
    # Enabled model with TODO model_id must fail
    with pytest.raises(ValidationError) as exc_info:
        ModelEntryConfig(
            name="test-todo-id",
            provider="openai_compatible",
            model_id="TODO_FILL_FROM_PROVIDER_DOCS",
            api_key_env="TEST_KEY",
            price_per_1k_input=0.001,
            price_per_1k_output=0.002,
            enabled=True,
        )
    assert "placeholder" in str(exc_info.value).lower() or "todo" in str(exc_info.value).lower()

    # Enabled model with TODO price_per_1k_input must fail
    with pytest.raises(ValidationError) as exc_info:
        ModelEntryConfig(
            name="test-todo-price-in",
            provider="openai_compatible",
            model_id="valid-model-id",
            api_key_env="TEST_KEY",
            price_per_1k_input="TODO_FILL_FROM_PROVIDER_DOCS",
            price_per_1k_output=0.002,
            enabled=True,
        )
    assert "placeholder" in str(exc_info.value).lower() or "todo" in str(exc_info.value).lower()

    # Enabled model with TODO price_per_1k_output must fail
    with pytest.raises(ValidationError) as exc_info:
        ModelEntryConfig(
            name="test-todo-price-out",
            provider="openai_compatible",
            model_id="valid-model-id",
            api_key_env="TEST_KEY",
            price_per_1k_input=0.001,
            price_per_1k_output="TODO_FILL_FROM_PROVIDER_DOCS",
            enabled=True,
        )
    assert "placeholder" in str(exc_info.value).lower() or "todo" in str(exc_info.value).lower()

    # Disabled model with TODO must succeed
    entry = ModelEntryConfig(
        name="test-disabled-todo",
        provider="openai_compatible",
        model_id="TODO_FILL_FROM_PROVIDER_DOCS",
        api_key_env="TEST_KEY",
        price_per_1k_input="TODO_FILL_FROM_PROVIDER_DOCS",
        price_per_1k_output="TODO_FILL_FROM_PROVIDER_DOCS",
        enabled=False,
    )
    assert entry.enabled is False


def test_models_yaml_extra_fields_forbidden():
    """Verify that models.yaml schema fails fast on unknown/extra fields."""
    with pytest.raises(ValidationError):
        ModelEntryConfig(
            name="test-extra",
            provider="openai_compatible",
            model_id="valid-model",
            api_key_env="TEST_KEY",
            price_per_1k_input=0.001,
            price_per_1k_output=0.002,
            enabled=True,
            unknown_extra_attribute="not_allowed",
        )

    with pytest.raises(ValidationError):
        ModelsConfig(
            default_model="glm-isoquant",
            unexpected_top_level_key=123,
        )


# ---------------------------------------------------------------------------
# 2. Missing Environment Variable Error Message
# ---------------------------------------------------------------------------
def test_missing_env_var_error_names_variable(temp_cache, temp_db, monkeypatch):
    """Verify missing API key raises error naming the variable, not any secret value."""
    var_name = "SPECIAL_NON_EXISTENT_API_KEY_VAR"
    monkeypatch.delenv(var_name, raising=False)

    cfg = ModelsConfig(
        mock_mode=False,
        cache_dir=temp_cache,
        models=[
            ModelEntryConfig(
                name="test-missing-key-model",
                provider="openai_compatible",
                model_id="glm-5.3-flash",
                base_url="https://api.isoquant.ai/v1",
                api_key_env=var_name,
                price_per_1k_input=0.001,
                price_per_1k_output=0.002,
                enabled=True,
            )
        ],
    )
    client = LLMClient(config=cfg, mock_mode=False, cache_dir=temp_cache, db_path=temp_db)

    with pytest.raises(ValueError) as exc_info:
        client.generate("Test prompt", model="test-missing-key-model", use_cache=False)

    err_msg = str(exc_info.value)
    assert var_name in err_msg, f"Error message should mention '{var_name}', got: {err_msg}"


# ---------------------------------------------------------------------------
# 3. Disk Cache Hit / Miss
# ---------------------------------------------------------------------------
def test_disk_cache_hit_and_miss(mock_client, temp_cache):
    """Verify disk caching: first call is miss, second is hit with cost=0 and cache_hit=true."""
    prompt = "Cached query about lead times"

    # 1. First call: cache miss
    resp1 = mock_client.generate(prompt=prompt, temperature=0.1, seed=123, use_cache=True)
    assert resp1.cached is False
    assert len(list(temp_cache.glob("*.json"))) == 1

    # 2. Second call: cache hit
    resp2 = mock_client.generate(prompt=prompt, temperature=0.1, seed=123, use_cache=True)
    assert resp2.cached is True
    assert resp2.content == resp1.content
    assert resp2.cost_usd == 0.0

    # 3. Different parameters: cache miss
    resp3 = mock_client.generate(prompt=prompt, temperature=0.2, seed=999, use_cache=True)
    assert resp3.cached is False
    assert len(list(temp_cache.glob("*.json"))) == 2


# ---------------------------------------------------------------------------
# 4. Retry on 429 Rate Limit
# ---------------------------------------------------------------------------
def test_retry_on_429_success(temp_cache, temp_db):
    """Verify retries on 429 with exponential backoff succeed when transient."""
    cfg = ModelsConfig(
        mock_mode=True,
        cache_dir=temp_cache,
        models=[
            ModelEntryConfig(
                name="mock",
                provider="mock",
                model_id="mock-model",
                api_key_env="MOCK_API_KEY",
                price_per_1k_input=0.0,
                price_per_1k_output=0.0,
                enabled=True,
            )
        ],
    )
    client = LLMClient(config=cfg, mock_mode=True, cache_dir=temp_cache, db_path=temp_db)

    # Simulate 429 on first 2 calls, then succeed on 3rd
    rate_err = Exception("429 Too Many Requests: Rate limit exceeded")
    client.mock.set_error("simulate_rate_limit", rate_err, fail_count=2)

    resp = client.generate("simulate_rate_limit: test retry", use_cache=False)
    assert resp.content is not None
    # Mock history records all 3 calls (initial attempt + 2 retries)
    assert len(client.mock.call_history) == 3


def test_retry_on_429_exhausted(temp_cache, temp_db):
    """Verify that if retries are exhausted on 429, the error is propagated."""
    cfg = ModelsConfig(
        mock_mode=True,
        cache_dir=temp_cache,
        max_retries=2,
        models=[
            ModelEntryConfig(
                name="mock",
                provider="mock",
                model_id="mock-model",
                api_key_env="MOCK_API_KEY",
                price_per_1k_input=0.0,
                price_per_1k_output=0.0,
                enabled=True,
            )
        ],
    )
    client = LLMClient(config=cfg, mock_mode=True, cache_dir=temp_cache, db_path=temp_db)

    rate_err = Exception("429 Rate limit exceeded permanently")
    client.mock.set_error("simulate_exhausted_429", rate_err, fail_count=5)

    with pytest.raises(Exception) as exc_info:
        client.generate("simulate_exhausted_429: prompt", use_cache=False)
    assert "429" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 5. Structured Output Retry & StructuredOutputError
# ---------------------------------------------------------------------------
def test_structured_output_retry_success(mock_client):
    """Verify structured output retries once on validation failure and succeeds if retry matches schema."""
    # Attempt 1 returns invalid json; attempt 2 returns valid json
    invalid_json = "This is not valid json at all!"
    valid_json = SampleDecisionModel(
        recommended_order=250.0,
        rationale="Recovered via structured retry",
        risk_level="medium",
    ).model_dump_json()

    mock_client.mock.set_sequential_responses("retry_success_prompt", [invalid_json, valid_json])

    resp = mock_client.generate(
        prompt="retry_success_prompt: provide replenishment",
        response_model=SampleDecisionModel,
        use_cache=False,
    )

    assert resp.structured is not None
    assert resp.structured.recommended_order == 250.0
    assert resp.structured.risk_level == "medium"
    assert resp.structured_output_ok is True
    assert resp.retry_count == 1


def test_structured_output_retry_failure_raises_error(mock_client, temp_db):
    """Verify structured output raises StructuredOutputError when both initial and retry fail."""
    invalid_1 = "Invalid JSON 1"
    invalid_2 = "Invalid JSON 2"

    mock_client.mock.set_sequential_responses("retry_failure_prompt", [invalid_1, invalid_2])

    with pytest.raises(StructuredOutputError) as exc_info:
        mock_client.generate(
            prompt="retry_failure_prompt: produce structured",
            response_model=SampleDecisionModel,
            use_cache=False,
            run_id="run_struct_fail",
        )

    assert "Structured output validation failed" in str(exc_info.value)

    # Check DuckDB recorded the failure
    conn = duckdb.connect(str(temp_db), read_only=True)
    rows = conn.execute(
        """
        SELECT structured_output_ok, error_type, retry_count
        FROM agent_calls
        WHERE run_id = 'run_struct_fail'
        """
    ).fetchall()
    conn.close()

    assert len(rows) == 1
    struct_ok, err_type, retries = rows[0]
    assert struct_ok is False
    assert err_type == "StructuredOutputError"
    assert retries == 1


# ---------------------------------------------------------------------------
# 6. Cost Calculation from Entry Prices
# ---------------------------------------------------------------------------
def test_cost_calculation_from_prices(mock_client):
    """Verify cost calculation derives from model entry prices."""
    # glm-isoquant: price_per_1k_input 0.000154, price_per_1k_output 0.00044
    cost = calculate_cost("glm-isoquant", prompt_tokens=1000, completion_tokens=1000, config=mock_client.config)
    # Expected: (1000/1000 * 0.000154) + (1000/1000 * 0.00044) = 0.000594
    assert cost == pytest.approx(0.000594)

    # 10,000 prompt tokens, 5,000 completion tokens:
    # (10 * 0.000154) + (5 * 0.00044) = 0.00154 + 0.00220 = 0.00374
    cost_large = calculate_cost("glm-isoquant", prompt_tokens=10000, completion_tokens=5000, config=mock_client.config)
    assert cost_large == pytest.approx(0.00374)

    # Fallback to standard table for gpt-4o-mini ($0.15/1M in, $0.60/1M out)
    cost_mini = calculate_cost("gpt-4o-mini", 1_000_000, 1_000_000)
    assert cost_mini == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# 7. Security: No Key Value Ever Appears in Logs
# ---------------------------------------------------------------------------
def test_no_api_key_in_logs(caplog, temp_cache, temp_db, monkeypatch):
    """Assert that private API key values never appear in any log output."""
    secret_key = "super-secret-isoquant-api-key-999-never-leak"
    monkeypatch.setenv("ISOQUANT_API_KEY", secret_key)

    cfg = ModelsConfig(
        mock_mode=True,
        cache_dir=temp_cache,
        models=[
            ModelEntryConfig(
                name="glm-isoquant",
                provider="openai_compatible",
                model_id="glm-5.3-flash",
                base_url="https://api.isoquant.ai/v1",
                api_key_env="ISOQUANT_API_KEY",
                price_per_1k_input=0.000154,
                price_per_1k_output=0.00044,
                enabled=True,
            )
        ],
    )
    client = LLMClient(config=cfg, mock_mode=True, cache_dir=temp_cache, db_path=temp_db)

    with caplog.at_level(logging.DEBUG):
        # Trigger normal call
        client.generate("Test log safety prompt", model="glm-isoquant", use_cache=False)
        # Trigger error call
        client.mock.set_error("simulate_log_error", Exception("Simulated network failure"), fail_count=1)
        try:
            client.generate("simulate_log_error: trigger failure", model="glm-isoquant", use_cache=False)
        except Exception:
            pass

    # Assert secret never appears in captured log text
    assert secret_key not in caplog.text


# ---------------------------------------------------------------------------
# 8. Tokenizer Fallback for Missing Token Counts
# ---------------------------------------------------------------------------
def test_tokenizer_fallback_when_tokens_missing(mock_client, temp_db):
    """Verify tokenizer fallback estimates tokens and sets usage_estimated=True when provider reports 0 tokens."""
    mock_client.mock.simulate_missing_tokens = True
    prompt = "Inventory replenishment forecast for next 14 days"

    resp = mock_client.generate(prompt=prompt, use_cache=False, run_id="run_fallback_tok")
    assert resp.usage_estimated is True
    assert resp.prompt_tokens > 0
    assert resp.completion_tokens > 0

    conn = duckdb.connect(str(temp_db), read_only=True)
    rows = conn.execute(
        """
        SELECT prompt_tokens, completion_tokens, usage_estimated
        FROM agent_calls
        WHERE run_id = 'run_fallback_tok'
        """
    ).fetchall()
    conn.close()

    assert len(rows) == 1
    p_tok, c_tok, est = rows[0]
    assert p_tok > 0
    assert c_tok > 0
    assert est is True


# ---------------------------------------------------------------------------
# 9. Deterministic Mock Double Features
# ---------------------------------------------------------------------------
def test_mock_llm_deterministic_text(mock_client):
    """Test MockLLM returns deterministic text output and logs history."""
    prompt = "Analyze inventory risk for retailer"
    resp: LLMResponse = mock_client.generate(prompt=prompt, temperature=0.2, seed=42)

    assert resp.content is not None
    assert "prompt_len=" in resp.content
    assert resp.prompt_tokens > 0
    assert resp.completion_tokens > 0
    assert resp.total_tokens == resp.prompt_tokens + resp.completion_tokens

    assert len(mock_client.mock.call_history) == 1
    call = mock_client.mock.call_history[0]
    assert call["prompt"] == prompt
    assert call["temperature"] == 0.2
    assert call["seed"] == 42


def test_mock_llm_canned_responses(mock_client):
    """Test registering and matching canned responses in MockLLM."""
    mock_client.mock.set_response("urgent order", "Order 500 units immediately.")

    resp1 = mock_client.generate("We have an urgent order incoming!")
    assert resp1.content == "Order 500 units immediately."

    resp2 = mock_client.generate("Standard review")
    assert "prompt_len=" in resp2.content


def test_mock_llm_structured_output(mock_client):
    """Test Pydantic structured output generation through MockLLM."""
    prompt = "Determine replenishment decision"
    resp: LLMResponse[SampleDecisionModel] = mock_client.generate(
        prompt=prompt,
        response_model=SampleDecisionModel,
    )

    assert resp.structured is not None
    assert isinstance(resp.structured, SampleDecisionModel)
    assert resp.structured.recommended_order == 100.0
    assert resp.structured.risk_level == "low"


def test_mock_llm_canned_structured_output(mock_client):
    """Test registering a custom structured model response in MockLLM."""
    custom = SampleDecisionModel(
        recommended_order=320.0,
        rationale="Buffer against supplier lead-time delay",
        risk_level="high",
    )
    mock_client.mock.set_response("supplier disruption", custom)

    resp = mock_client.generate(
        prompt="Handling supplier disruption scenario",
        response_model=SampleDecisionModel,
    )
    assert resp.structured is not None
    assert resp.structured.recommended_order == 320.0
    assert resp.structured.risk_level == "high"
    assert "Buffer against" in resp.structured.rationale


def test_duckdb_agent_calls_logging(mock_client, temp_db):
    """Verify LLM calls log latency, tokens, cost, and provider metadata to DuckDB."""
    mock_client.generate(
        prompt="Order recommendation for wholesaler",
        agent_name="inventory_agent",
        run_id="run_test_123",
        step_id="step_5",
        use_cache=False,
    )

    conn = duckdb.connect(str(temp_db), read_only=True)
    rows = conn.execute(
        """
        SELECT agent_name, model_name, provider, prompt_tokens, completion_tokens, latency_ms
        FROM agent_calls
        WHERE run_id = 'run_test_123'
        """
    ).fetchall()
    conn.close()

    assert len(rows) == 1
    agent_name, model_name, prov, p_tok, c_tok, lat = rows[0]
    assert agent_name == "inventory_agent"
    assert model_name == "mock"
    assert prov == "mock"
    assert p_tok > 0
    assert c_tok > 0
    assert lat >= 0.0


def test_opik_tracing_flag(temp_cache, temp_db):
    """Verify client handles opik_enabled flag gracefully without crash."""
    cfg = ModelsConfig(
        mock_mode=True,
        opik_enabled=True,
        cache_dir=temp_cache,
    )
    client = LLMClient(config=cfg, mock_mode=True, cache_dir=temp_cache, db_path=temp_db)
    resp = client.generate("Test prompt with Opik enabled")
    assert resp.content is not None
