"""Tests for the Systems 1-6 results page (src/sca/dashboard/results.py)."""

import json

import altair as alt
import pytest
from streamlit.testing.v1 import AppTest

from sca.dashboard import results
from sca.dashboard.results import DATASETS, S6, load_results


@pytest.fixture
def data_dir(tmp_path):
    compare = [
        {"system": "S1 (s,S) policy", "kind": "Traditional", "status": "Ran", "total_cost": 32000, "fill_rate": 0.97,
         "e2e_bullwhip": 4.0, "bw_retailer": 1.2, "bw_wholesaler": 1.1, "bw_distributor": 1.0, "bw_manufacturer": 1.0,
         "stockout_steps": 20, "latency_ms": 0.1},
        {"system": "S2 EOQ", "kind": "Traditional", "status": "Ran", "total_cost": 26000, "fill_rate": 1.0,
         "e2e_bullwhip": 3.9, "bw_retailer": 1.8, "bw_wholesaler": 1.2, "bw_distributor": 1.2, "bw_manufacturer": 1.1,
         "stockout_steps": 17, "latency_ms": 0.1},
        {"system": "S4 Multi-agent + info sharing", "kind": "LLM", "status": "Not run with a real LLM",
         "total_cost": None, "fill_rate": None, "e2e_bullwhip": None, "bw_retailer": None, "bw_wholesaler": None,
         "bw_distributor": None, "bw_manufacturer": None, "stockout_steps": None, "latency_ms": None},
        {"system": S6, "kind": "LLM", "status": "Ran", "total_cost": 26200, "fill_rate": 1.0, "e2e_bullwhip": 4.27,
         "bw_retailer": 4.4, "bw_wholesaler": 1.0, "bw_distributor": 0.8, "bw_manufacturer": 0.8,
         "stockout_steps": None, "latency_ms": None},
    ]
    bw = [{"system": r["system"], "echelon": e, "bullwhip": r[f"bw_{e}"]} for r in compare
          for e in ("retailer", "wholesaler", "distributor", "manufacturer") if r["e2e_bullwhip"] is not None]
    data = {
        "compare": compare, "bw_echelon": bw,
        "behaviour": [{"system": S6, "instruction_following": 1.0, "fallback_rate": 0.0, "error_rate": 0.0,
                       "regime_accuracy": 0.74, "follow_rate": 0.59, "clip_rate": 0.64, "tokens_in": 838,
                       "tokens_out": 85, "usd_per_run": 0.0099, "groundedness": None, "latency_ms": None,
                       "tool_choice_accuracy": "n/a", "parameter_accuracy": "n/a", "negotiation": "n/a"}],
        "action_mix": [{"system": S6, "echelon": "retailer", "follow": 0.6, "boost_small": 0.1, "boost_large": 0.0,
                        "trim": 0.3, "fallback": 0.0}],
        "earlier": [{"system": "S6 Orchestrated", "prompt_version": "v3.1", "seed": 100, "latency_ms": 10240.0,
                     "valid_rate": 1.0, "fallbacks": 0, "rounds": 1.76, "messages": 3.5, "tokens_in": 665,
                     "tokens_out": 1691, "usd": 0.24, "note": "Valid"}],
        "baselines_test": [{"system": "S2 EOQ", "scenario": "shock", "total_cost": 29000, "fill_rate": 0.97,
                            "e2e_bullwhip": 2.7, "stockout_steps": 5.0, "runs": 8, "key": "S2 EOQ|shock"}],
    }
    for key, name in DATASETS.items():
        (tmp_path / name).write_text(json.dumps(data[key]), encoding="utf-8")
    return tmp_path


def test_load_results_and_headline(data_dir):
    frames = load_results(data_dir)
    assert set(frames) == set(DATASETS)
    cmp_ = frames["compare"]
    assert cmp_.loc[cmp_.system == S6, "group"].item() == "Proposed"
    assert not cmp_.loc[cmp_.system.str.startswith("S4"), "ran"].item()
    h = results.headline(cmp_)
    assert h["s6_total_cost"] == 26200 and h["eoq_total_cost"] == 26000
    assert h["best_bw_system"] == "S2 EOQ"


def test_missing_files_give_empty_frames(tmp_path):
    frames = load_results(tmp_path)
    assert all(df.empty for df in frames.values())


@pytest.mark.parametrize("metric", list(results.PERF_METRICS))
def test_charts_compile(data_dir, metric):
    frames = load_results(data_dir)
    for chart in (results.performance_bar(frames["compare"], metric), results.echelon_lines(frames["bw_echelon"]),
                  results.action_mix_bars(frames["action_mix"]),
                  results.baselines_chart(frames["baselines_test"], "total_cost")):
        assert isinstance(chart.to_dict(), dict)


def test_page_renders_without_errors(data_dir):
    def app(path):
        from sca.dashboard.results import render_results_page

        render_results_page(path)

    at = AppTest.from_function(app, args=(str(data_dir),), default_timeout=60).run()
    assert not at.exception
    assert at.title[0].value == "Systems 1–6: results"
    assert len(at.metric) == 4
    assert len(at.tabs) == 7


def test_page_handles_missing_data(tmp_path):
    def app(path):
        from sca.dashboard.results import render_results_page

        render_results_page(path)

    at = AppTest.from_function(app, args=(str(tmp_path),), default_timeout=60).run()
    assert not at.exception
    assert at.warning


def test_behaviour_table_is_metrics_by_system(data_dir):
    t = results.behaviour_table(load_results(data_dir)["behaviour"])
    assert list(t.columns) == ["S6 Orchestrated"]
    assert t.loc["Instruction following", "S6 Orchestrated"] == "100%"
    assert t.loc["Groundedness", "S6 Orchestrated"] == "not logged"
    assert t.loc["USD per run", "S6 Orchestrated"] == "$0.0099"


@pytest.fixture
def model_runs(tmp_path):
    rows = []
    for model in ("glm-isoquant", "deepseek-v4.1-flash", "gemini-3.8-flash"):
        for system in ("S3 single agent", "S6 orchestrated"):
            for scen in ("stable", "shock"):
                rows.append({"model": model, "system": system, "scenario": scen, "seed": 102, "total_cost": 25000.0,
                             "fill_rate": 100.0, "e2e_bullwhip": 4.2, "stockout_steps": 3,
                             "tool_choice_accuracy": 1.0 if system.startswith("S6") else None,
                             "parameter_accuracy": 0.9 if system.startswith("S6") else None,
                             "instruction_following": 1.0, "groundedness": 0.8, "fallback_rate": 0.0,
                             "error_rate": 0.0, "negotiation_rounds": 1.5 if system.startswith("S6") else None,
                             "negotiation_messages": 6.0 if system.startswith("S6") else None,
                             "tokens_in": 800, "tokens_out": 50, "usd": 0.01, "llm_latency_ms": 2500,
                             "invalid_reason": None})
    rows.append({"model": "deterministic", "system": "S2 EOQ tuned", "scenario": "shock", "seed": 102,
                 "total_cost": 26000.0, "fill_rate": 100.0, "e2e_bullwhip": 3.9, "stockout_steps": 17})
    path = tmp_path / "runs.csv"
    import pandas as pd

    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_model_summary_and_chart(model_runs):
    runs = results.load_model_runs(model_runs)
    s = results.model_summary(runs[runs.model != "deterministic"], ["shock"])
    assert set(s.model_label) == {"GLM-5.3 Flash", "DeepSeek V4.1 Flash", "Gemini 3.8 Flash"}
    assert (s.runs == 1).all() and (s.invalid_runs == 0).all()
    row = s[(s.model_label == "GLM-5.3 Flash") & (s.system_label == "S6 Orchestrated")].iloc[0]
    assert row.negotiation_rounds == 1.5 and row.fill_rate == 1.0
    for metric in results.MODEL_METRICS:
        assert isinstance(results.model_bars(results.model_summary(runs), metric).to_dict(), dict)


def test_page_renders_model_tab(data_dir, model_runs):
    def app(path, runs_path):
        from sca.dashboard.results import render_results_page

        render_results_page(path, runs_path)

    at = AppTest.from_function(app, args=(str(data_dir), str(model_runs)), default_timeout=60).run()
    assert not at.exception
    assert len(at.tabs) == 7
    assert at.tabs[0].label == "Model comparison"


def test_coverage_marks_unrun_cells(model_runs):
    cov = results.coverage(results.load_model_runs(model_runs)).set_index("Model")
    assert cov.loc["GLM-5.3 Flash", "S3 Single agent"] == "2/3 measured"
    assert cov.loc["GLM-5.3 Flash", "S4 Multi-agent + info"] == "not run"


def test_headline_from_runs(model_runs):
    runs = results.load_model_runs(model_runs)
    h = results.headline_from_runs(runs, "shock")
    assert h["s6_total_cost"] == 25000.0 and h["eoq_total_cost"] == 26000.0
    assert h["best_bw_system"] == "S2 EOQ"
    assert results.headline_from_runs(runs, "seasonal") == {}
