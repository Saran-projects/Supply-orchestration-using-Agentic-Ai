"""Results page: Systems 1-6 comparison (supply chain performance and agent behaviour).

Reads the datasets written by ``scripts/dashboard_existing_data.py`` from
``<experiment.output_dir>/dashboard/existing`` (configs/experiment.yaml). Read-only; no LLM calls.
Charts are Altair (bundled with Streamlit): hover tooltips, click-to-highlight legends, and
layouts that stack on narrow screens.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import altair as alt
import pandas as pd
import streamlit as st

from sca.config import find_project_root, get_config

ECHELONS = ["retailer", "wholesaler", "distributor", "manufacturer"]
S6 = "S6 Orchestrated (proposed)"
EOQ = "S2 EOQ"
DATASETS = {
    "compare": "compare_seed102.json",
    "bw_echelon": "bullwhip_echelon_seed102.json",
    "behaviour": "behaviour_v5.json",
    "action_mix": "action_mix_v5.json",
    "earlier": "behaviour_earlier.json",
    "baselines_test": "baselines_test_seeds.json",
}
# Performance metrics: column, label, format, whether lower is better
PERF_METRICS = {
    "total_cost": ("Total cost", ",.0f", True),
    "e2e_bullwhip": ("End-to-end bullwhip", ".2f", True),
    "fill_rate": ("Fill rate", ".1%", False),
    "stockout_steps": ("Stockouts", ".0f", True),
    "latency_ms": ("Response time (ms)", ",.2f", True),
}
KIND_COLORS = {"Traditional": "#64748B", "LLM": "#94A3B8", "Reference": "#CBD5E1", "Proposed": "#0E7C66"}
ACTION_COLORS = {"follow": "#0E7C66", "boost_small": "#5B8DEF", "boost_large": "#2F4FA8", "trim": "#D9822B",
                 "fallback": "#C2410C"}


# ---------------------------------------------------------------------------
# Data (pure, testable)
# ---------------------------------------------------------------------------
def default_data_dir() -> Path:
    out = Path(get_config().experiment.output_dir)
    if not out.is_absolute():
        out = find_project_root() / out
    return out / "dashboard" / "existing"


def load_results(data_dir: Optional[Path | str] = None) -> Dict[str, pd.DataFrame]:
    """Load every dataset as a DataFrame; a missing file gives an empty frame."""
    base = Path(data_dir) if data_dir else default_data_dir()
    frames: Dict[str, pd.DataFrame] = {}
    for key, name in DATASETS.items():
        path = base / name
        frames[key] = pd.DataFrame(json.loads(path.read_text(encoding="utf-8"))) if path.exists() else pd.DataFrame()
    cmp_ = frames["compare"]
    if not cmp_.empty:
        cmp_["group"] = cmp_.apply(lambda r: "Proposed" if r["system"] == S6 else r["kind"], axis=1)
        cmp_["ran"] = cmp_["total_cost"].notna()
    return frames


def headline(compare: pd.DataFrame) -> Dict[str, Optional[float]]:
    """S6 against S2 EOQ, and the system with the lowest end-to-end bullwhip."""
    out: Dict[str, Optional[float]] = {}
    if compare.empty:
        return out
    rows = compare.set_index("system")
    for key in ("total_cost", "e2e_bullwhip", "fill_rate"):
        out[f"s6_{key}"] = rows.at[S6, key] if S6 in rows.index else None
        out[f"eoq_{key}"] = rows.at[EOQ, key] if EOQ in rows.index else None
    ran = compare[compare["e2e_bullwhip"].notna()]
    best = ran.loc[ran["e2e_bullwhip"].idxmin()]
    out["best_bw_system"] = best["system"]
    out["best_bw"] = best["e2e_bullwhip"]
    return out


def short(name: str) -> str:
    return name.replace(" (proposed)", "").replace("Ref: ", "")


BEHAVIOUR_ROWS = [
    ("instruction_following", "Instruction following", ".0%"),
    ("fallback_rate", "Fallback rate", ".0%"),
    ("error_rate", "Error rate", ".0%"),
    ("regime_accuracy", "Regime accuracy", ".0%"),
    ("follow_rate", 'Chose "follow"', ".0%"),
    ("clip_rate", "Clipped to safe range", ".0%"),
    ("tokens_in", "Tokens in per decision", ",.0f"),
    ("tokens_out", "Tokens out per decision", ",.0f"),
    ("usd_per_run", "USD per run", "$.4f"),
    ("groundedness", "Groundedness", ".0%"),
    ("latency_ms", "Response time", ",.0f"),
    ("tool_choice_accuracy", "Tool-choice accuracy", ""),
    ("parameter_accuracy", "Parameter accuracy", ""),
    ("negotiation", "Negotiation", ""),
]


def behaviour_table(b: pd.DataFrame) -> pd.DataFrame:
    """Metrics as rows, systems as columns, values formatted as text."""
    out = {}
    for _, row in b.iterrows():
        col = []
        for field, _label, spec in BEHAVIOUR_ROWS:
            v = row.get(field)
            if v is None or (isinstance(v, float) and pd.isna(v)):
                col.append("not logged")
            elif isinstance(v, str):
                col.append(v)
            elif spec.startswith("$"):
                col.append("$" + format(v, spec[1:]))
            else:
                col.append(format(v, spec))
        out[short(row["system"])] = col
    return pd.DataFrame(out, index=[label for _f, label, _s in BEHAVIOUR_ROWS])


# ---------------------------------------------------------------------------
# Model comparison (design v5.1, protocol Amendment A2)
# ---------------------------------------------------------------------------
MODEL_LABELS = {"glm-isoquant": "GLM-5.3 Flash", "deepseek-v4.1-flash": "DeepSeek V4.1 Flash",
                "gemini-3.8-flash": "Gemini 3.8 Flash", "deterministic": "Deterministic"}
MODEL_COLORS = {"GLM-5.3 Flash": "#0E7C66", "DeepSeek V4.1 Flash": "#5B8DEF", "Gemini 3.8 Flash": "#D9822B",
                "Deterministic": "#94A3B8"}
SYSTEM_LABELS = {"S1 (s,S) tuned": "S1 (s,S) policy", "S2 EOQ tuned": "S2 EOQ", "ref 2c tuned": "Ref: 2c heuristic",
                 "S3 single agent": "S3 Single agent", "S4 multi-agent info": "S4 Multi-agent + info",
                 "S5 multi-agent tools": "S5 Multi-agent + tools", "S6 orchestrated": "S6 Orchestrated"}
SYSTEM_ORDER = list(SYSTEM_LABELS.values())
# column -> (label, d3 format, lower is better, which systems it applies to)
MODEL_METRICS = {
    "total_cost": ("Total cost", ",.0f", True, "all"),
    "e2e_bullwhip": ("End-to-end bullwhip", ".2f", True, "all"),
    "fill_rate": ("Fill rate", ".1%", False, "all"),
    "stockout_steps": ("Stockout steps", ".1f", True, "all"),
    "tool_choice_accuracy": ("Tool-choice accuracy", ".0%", False, "S5, S6"),
    "parameter_accuracy": ("Parameter accuracy", ".0%", False, "S5, S6"),
    "instruction_following": ("Instruction following", ".0%", False, "LLM"),
    "groundedness": ("Groundedness", ".0%", False, "LLM"),
    "fallback_rate": ("Fallback rate", ".1%", True, "LLM"),
    "error_rate": ("Error rate", ".1%", True, "LLM"),
    "negotiation_rounds": ("Negotiation rounds per decision", ".2f", True, "S6"),
    "negotiation_messages": ("Messages per decision", ".2f", True, "S6"),
    "tokens_in": ("Tokens in per decision", ",.0f", True, "LLM"),
    "tokens_out": ("Tokens out per decision", ",.0f", True, "LLM"),
    "usd": ("USD per run", "$.4f", True, "LLM"),
    "llm_latency_ms": ("Response time per decision (ms)", ",.0f", True, "LLM"),
}
PERF_KEYS = ["total_cost", "e2e_bullwhip", "fill_rate", "stockout_steps"]
PLANNED_MODELS = ["glm-isoquant", "deepseek-v4.1-flash", "gemini-3.8-flash"]
LLM_SYSTEMS = ["S3 single agent", "S4 multi-agent info", "S5 multi-agent tools", "S6 orchestrated"]
SCENARIOS = ["stable", "seasonal", "shock"]


def coverage(runs: pd.DataFrame) -> pd.DataFrame:
    """Measured runs out of planned (3 scenarios) per model and system; nothing is estimated."""
    done = set(zip(runs["model"], runs["system"], runs["scenario"])) if not runs.empty else set()
    out = []
    for m in PLANNED_MODELS:
        row = {"Model": MODEL_LABELS[m]}
        for sys_ in LLM_SYSTEMS:
            n = sum((m, sys_, sc) in done for sc in SCENARIOS)
            row[SYSTEM_LABELS[sys_]] = f"{n}/3 measured" if n else "not run"
        out.append(row)
    return pd.DataFrame(out)


def default_model_runs_path() -> Path:
    out = Path(get_config().experiment.output_dir)
    if not out.is_absolute():
        out = find_project_root() / out
    return out / "v51_model_runs.csv"


def load_model_runs(path: Optional[Path | str] = None) -> pd.DataFrame:
    """One row per (model, system, scenario) run; deterministic references have model 'deterministic'."""
    p = Path(path) if path else default_model_runs_path()
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p).drop_duplicates(["model", "system", "scenario"], keep="last")
    df["fill_rate"] = df["fill_rate"] / 100.0
    df["model_label"] = df["model"].map(MODEL_LABELS).fillna(df["model"])
    df["system_label"] = df["system"].map(SYSTEM_LABELS).fillna(df["system"])
    return df


def headline_from_runs(runs: pd.DataFrame, scenario: str, model: str = "glm-isoquant") -> Dict[str, Optional[float]]:
    """S6 (given model) against S2 EOQ on one scenario, from the model-comparison runs; {} if either is missing."""
    if runs.empty:
        return {}
    df = runs[(runs["scenario"] == scenario) & runs["model"].isin([model, "deterministic"])]
    s6 = df[(df["model"] == model) & (df["system"] == "S6 orchestrated")]
    eoq = df[df["system"] == "S2 EOQ tuned"]
    if s6.empty or eoq.empty:
        return {}
    out: Dict[str, Optional[float]] = {}
    for key in ("total_cost", "e2e_bullwhip", "fill_rate"):
        out[f"s6_{key}"] = float(s6[key].iloc[0])
        out[f"eoq_{key}"] = float(eoq[key].iloc[0])
    best = df.loc[df["e2e_bullwhip"].idxmin()]
    out["best_bw_system"] = best["system_label"]
    out["best_bw"] = float(best["e2e_bullwhip"])
    return out


def model_summary(runs: pd.DataFrame, scenarios: Optional[List[str]] = None) -> pd.DataFrame:
    """Mean of each metric per (model, system) over the chosen scenarios."""
    if runs.empty:
        return runs
    df = runs if not scenarios else runs[runs["scenario"].isin(scenarios)]
    cols = [c for c in MODEL_METRICS if c in df.columns]
    keys = ["model_label", "system_label"]
    out = df.groupby(keys, as_index=False)[cols].mean(numeric_only=True)
    out = out.merge(df.groupby(keys).size().rename("runs").reset_index(), on=keys)
    if "invalid_reason" in df.columns:
        inv = df.assign(_inv=df["invalid_reason"].notna().astype(int)).groupby(keys)["_inv"].sum()
        out = out.merge(inv.rename("invalid_runs").reset_index(), on=keys)
    return out


def model_bars(summary: pd.DataFrame, metric: str) -> alt.Chart:
    label, fmt, lower_better, _scope = MODEL_METRICS[metric]
    data = summary[summary[metric].notna()]
    systems = [x for x in SYSTEM_ORDER if x in set(data["system_label"])]
    models = [m for m in MODEL_COLORS if m in set(data["model_label"])]
    d3fmt = fmt.lstrip("$")
    return alt.Chart(data).mark_bar(cornerRadiusEnd=3).encode(
        y=alt.Y("system_label:N", sort=systems, title=None, scale=alt.Scale(paddingInner=0.25),
                axis=alt.Axis(labelLimit=220)),
        yOffset=alt.YOffset("model_label:N", sort=models),
        x=alt.X(f"{metric}:Q", title=label, axis=alt.Axis(format=d3fmt)),
        color=alt.Color("model_label:N", sort=models,
                        scale=alt.Scale(domain=models, range=[MODEL_COLORS[m] for m in models]),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=[alt.Tooltip("model_label:N", title="Model"), alt.Tooltip("system_label:N", title="System"),
                 alt.Tooltip(f"{metric}:Q", title=label, format=d3fmt), alt.Tooltip("runs:Q", title="Runs")],
    ).properties(height=alt.Step(16), title=alt.Title(
        label, subtitle=("lower is better" if lower_better else "higher is better")))


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def _group_scale() -> alt.Scale:
    return alt.Scale(domain=list(KIND_COLORS), range=list(KIND_COLORS.values()))


def performance_bar(df: pd.DataFrame, metric: str) -> alt.Chart:
    label, fmt, lower_better = PERF_METRICS[metric]
    data = df[df[metric].notna()].copy()
    data["label"] = data["system"].map(short)
    order = data.sort_values(metric, ascending=lower_better)["label"].tolist()
    hover = alt.selection_point(on="pointerover", fields=["label"], empty=False)
    base = alt.Chart(data).encode(
        y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=220),
                scale=alt.Scale(paddingInner=0.3)),
        x=alt.X(f"{metric}:Q", title=label, axis=alt.Axis(format=fmt, grid=True)),
        tooltip=[alt.Tooltip("system:N", title="System"), alt.Tooltip(f"{metric}:Q", title=label, format=fmt),
                 alt.Tooltip("status:N", title="Status")],
    )
    bars = base.mark_bar(cornerRadiusEnd=4).encode(
        color=alt.Color("group:N", scale=_group_scale(), legend=alt.Legend(title=None, orient="bottom")),
        opacity=alt.condition(hover, alt.value(1), alt.value(0.85)),
        stroke=alt.condition(hover, alt.value("#0F172A"), alt.value(None)),
    ).add_params(hover)
    text = base.mark_text(align="left", dx=4, fontSize=11).encode(text=alt.Text(f"{metric}:Q", format=fmt))
    note = "lower is better" if lower_better else "higher is better"
    return (bars + text).properties(height=alt.Step(36), title=alt.Title(label, subtitle=note))


def echelon_lines(bw: pd.DataFrame) -> alt.Chart:
    data = bw[bw["bullwhip"].notna()].copy()
    data["label"] = data["system"].map(short)
    pick = alt.selection_point(fields=["label"], bind="legend")
    width = alt.condition(alt.datum.label == short(S6), alt.value(3.5), alt.value(1.8))
    base = alt.Chart(data).encode(
        x=alt.X("echelon:N", sort=ECHELONS, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("bullwhip:Q", title="Bullwhip ratio"),
        color=alt.Color("label:N", title="System", legend=alt.Legend(orient="bottom", columns=3)),
        opacity=alt.condition(pick, alt.value(1), alt.value(0.12)),
        tooltip=[alt.Tooltip("system:N", title="System"), alt.Tooltip("echelon:N", title="Echelon"),
                 alt.Tooltip("bullwhip:Q", title="Bullwhip", format=".2f")],
    )
    return (base.mark_line(strokeCap="round", strokeJoin="round").encode(strokeWidth=width)
            + base.mark_point(filled=True, size=60)).add_params(pick).properties(height=320)


def action_mix_bars(mix: pd.DataFrame) -> alt.Chart:
    long = mix.melt(id_vars=["system", "echelon"], value_vars=list(ACTION_COLORS), var_name="action", value_name="share")
    long["system"] = long["system"].map(short)
    pick = alt.selection_point(fields=["action"], bind="legend")
    return alt.Chart(long).mark_bar().encode(
        y=alt.Y("echelon:N", sort=ECHELONS, title=None, scale=alt.Scale(paddingInner=0.25)),
        x=alt.X("share:Q", stack="normalize", title="Share of decisions", axis=alt.Axis(format="%")),
        color=alt.Color("action:N", scale=alt.Scale(domain=list(ACTION_COLORS), range=list(ACTION_COLORS.values())),
                        legend=alt.Legend(orient="bottom", title=None)),
        opacity=alt.condition(pick, alt.value(1), alt.value(0.15)),
        row=alt.Row("system:N", title=None, header=alt.Header(labelAngle=0, labelAlign="left", labelFontWeight="bold")),
        tooltip=[alt.Tooltip("system:N", title="System"), alt.Tooltip("echelon:N", title="Echelon"),
                 alt.Tooltip("action:N", title="Action"), alt.Tooltip("share:Q", title="Share", format=".0%")],
    ).add_params(pick).properties(height=alt.Step(26))


def baselines_chart(df: pd.DataFrame, metric: str) -> alt.Chart:
    label, fmt, lower_better = {
        "total_cost": ("Total cost", ",.0f", True), "e2e_bullwhip": ("End-to-end bullwhip", ".2f", True),
        "fill_rate": ("Fill rate", ".1%", False), "stockout_steps": ("Stockout steps (retailer)", ".1f", True),
    }[metric]
    data = df.copy()
    data["label"] = data["system"].map(short)
    return alt.Chart(data).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X(f"{metric}:Q", title=label, axis=alt.Axis(format=fmt)),
        y=alt.Y("label:N", title=None, scale=alt.Scale(paddingInner=0.3)),
        color=alt.Color("label:N", legend=None),
        row=alt.Row("scenario:N", title=None, sort=["stable", "seasonal", "shock"],
                    header=alt.Header(labelAngle=0, labelAlign="left", labelFontWeight="bold")),
        tooltip=[alt.Tooltip("system:N", title="System"), alt.Tooltip("scenario:N", title="Scenario"),
                 alt.Tooltip(f"{metric}:Q", title=label, format=fmt), alt.Tooltip("runs:Q", title="Runs")],
    ).properties(height=alt.Step(26), title=alt.Title(label, subtitle="mean of 8 test seeds; " +
                                             ("lower is better" if lower_better else "higher is better")))


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def _fmt(v, spec: str, missing: str = "—") -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return missing
    return format(v, spec)


def render_model_comparison(path: Optional[Path | str] = None) -> None:
    runs = load_model_runs(path)
    if runs.empty:
        st.info("No model-comparison runs yet. Run `python scripts/v51_model_comparison.py --execute` to create them.")
        return
    llm = runs[runs["model"] != "deterministic"]
    st.markdown("**Coverage.** Only measured runs are shown anywhere on this tab; cells that were not run stay empty.")
    st.table(coverage(runs).set_index("Model"))
    st.caption("DeepSeek V4.1 Flash and Gemini 3.8 Flash run through OpenRouter. The run stopped on HTTP 402 "
               "(insufficient OpenRouter credit), so most of their cells were not run. They can be resumed with "
               "`scripts/v51_model_comparison.py --execute` once credit is added.")
    st.markdown("**Systems 3–6 on three LLMs, design v5.1, tuning seed 102.** Systems 5 and 6 choose their own tools "
                "and inputs, and System 6 negotiates with its specialists. One run per cell, so these are case "
                "results, not statistical findings. S1, S2 and the 2c heuristic appear in grey.")
    c1, c2 = st.columns([3, 2], vertical_alignment="bottom")
    group = c1.segmented_control("Show", ["Supply chain", "Agent behaviour"], default="Supply chain",
                                 key="mc_group") or "Supply chain"
    scen = c2.pills("Scenarios", ["stable", "seasonal", "shock"], selection_mode="multi",
                    default=["stable", "seasonal", "shock"], key="mc_scen") or ["stable", "seasonal", "shock"]
    keys = PERF_KEYS if group == "Supply chain" else [k for k in MODEL_METRICS if k not in PERF_KEYS]
    metric = st.selectbox("Metric", keys, key=f"mc_metric_{group}",
                          format_func=lambda k: f"{MODEL_METRICS[k][0]} (applies to {MODEL_METRICS[k][3]})")
    summary = model_summary(runs if group == "Supply chain" else llm, scen)
    if metric in summary and summary[metric].notna().any():
        st.altair_chart(model_bars(summary, metric), use_container_width=True)
    else:
        st.caption("No values for this metric yet.")

    by_model = model_summary(llm, scen)
    if by_model.empty:
        return
    st.markdown("**All metrics by model and system** (mean over the selected scenarios)")
    cols = ["model_label", "system_label", "runs"] + (["invalid_runs"] if "invalid_runs" in by_model else []) + \
        [k for k in MODEL_METRICS if k in by_model]
    cfg = {"model_label": "Model", "system_label": "System",
           "runs": st.column_config.NumberColumn("Runs", format="%d"),
           "invalid_runs": st.column_config.NumberColumn("Invalid runs", format="%d")}
    for k, (label, fmt, _lb, _sc) in MODEL_METRICS.items():
        if "%" in fmt:
            cfg[k] = st.column_config.NumberColumn(label, format="percent")
        elif fmt.startswith("$"):
            cfg[k] = st.column_config.NumberColumn(label, format="$%.4f")
        elif k == "total_cost":
            cfg[k] = st.column_config.NumberColumn(label, format="localized")
        else:
            cfg[k] = st.column_config.NumberColumn(label, format="%.2f")
    st.dataframe(by_model[cols].sort_values(["system_label", "model_label"]), hide_index=True,
                 use_container_width=True, placeholder="—", column_config=cfg)
    spend = llm[llm["scenario"].isin(scen)].groupby("model_label")["usd"].sum()
    st.caption("Spend over these runs: " + ", ".join(f"{m} ${v:.3f}" for m, v in spend.items()) +
               ". Tool-choice and parameter accuracy apply to S5 and S6 only, because S3 and S4 have no tools. "
               "Negotiation applies to S6 only. Groundedness is the share of numbers quoted in rationales that "
               "match a number in the prompt.")


def render_results_page(data_dir: Optional[Path | str] = None, model_runs_path: Optional[Path | str] = None) -> None:
    frames = load_results(data_dir)
    cmp_ = frames["compare"]
    st.title("Systems 1–6: results")
    if cmp_.empty:
        st.warning("No results found. Run `python scripts/dashboard_existing_data.py` to build them.")
        return
    runs = load_model_runs(model_runs_path)
    latest = headline_from_runs(runs, "shock")
    if latest:
        h = latest
        st.caption("Headline: shock scenario, tuning seed 102, newest glm runs (design v5.1), one 40-step run per "
                   "system. With one run per cell these are case results, not statistical findings.")
    else:
        h = headline(cmp_)
        st.caption("Shock scenario, seed 102, one 40-step run per system (5 warm-up steps excluded). "
                   "S3 and S6 used glm-5.3-flash with design v5.0. With one run per system these are case results, "
                   "not statistical findings.")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("S6 total cost", _fmt(h.get("s6_total_cost"), ",.0f"),
              delta=_fmt(h["s6_total_cost"] - h["eoq_total_cost"], "+,.0f") + " vs EOQ",
              delta_color="inverse", border=True)
    k2.metric("S6 end-to-end bullwhip", _fmt(h.get("s6_e2e_bullwhip"), ".2f"),
              delta=_fmt(h["s6_e2e_bullwhip"] - h["eoq_e2e_bullwhip"], "+.2f") + " vs EOQ",
              delta_color="inverse", border=True)
    k3.metric("S6 fill rate", _fmt(h.get("s6_fill_rate"), ".1%"), border=True)
    k4.metric("Lowest bullwhip", short(str(h.get("best_bw_system"))), delta=_fmt(h.get("best_bw"), ".2f"),
              delta_color="off", border=True)

    models_tab, perf, bw, beh, earlier, base, notes = st.tabs(
        ["Model comparison", "Performance", "Bullwhip by echelon", "Agent behaviour", "Earlier designs",
         "Baselines, test seeds", "Notes"])

    with models_tab:
        render_model_comparison(model_runs_path)

    with perf:
        c1, c2 = st.columns([1, 1], vertical_alignment="bottom")
        metric = c1.segmented_control("Metric", list(PERF_METRICS), default="total_cost",
                                      format_func=lambda m: PERF_METRICS[m][0], key="perf_metric") or "total_cost"
        systems = c2.pills("Systems", cmp_["system"].tolist(), selection_mode="multi",
                           default=cmp_["system"].tolist(), format_func=lambda n: short(n).split(" ")[0] if n.startswith("S") else short(n),
                           key="perf_systems") or []
        view = cmp_[cmp_["system"].isin(systems)]
        if view[metric].notna().any():
            st.altair_chart(performance_bar(view, metric), use_container_width=True)
        missing = view.loc[view[metric].isna(), "system"].map(short).tolist()
        if missing:
            st.caption(f"No value for: {', '.join(missing)} (not run, or not logged in that run).")
        st.dataframe(
            view[["system", "status", "total_cost", "fill_rate", "e2e_bullwhip", "stockout_steps", "latency_ms",
                  *[f"bw_{e}" for e in ECHELONS]]],
            hide_index=True, use_container_width=True, placeholder="—",
            column_config={
                "system": "System",
                "status": "Status",
                "total_cost": st.column_config.NumberColumn("Total cost", format="localized"),
                "fill_rate": st.column_config.ProgressColumn("Fill rate", format="percent", min_value=0, max_value=1),
                "e2e_bullwhip": st.column_config.NumberColumn("E2E bullwhip", format="%.2f"),
                "stockout_steps": st.column_config.NumberColumn("Stockout steps", format="%d"),
                "latency_ms": st.column_config.NumberColumn("Response time (ms)", format="%.2f"),
                **{f"bw_{e}": st.column_config.NumberColumn(f"BW {e}", format="%.2f") for e in ECHELONS},
            },
        )
        st.caption("Total cost = holding (0.5) + backlog (2.5) per unit-day, all echelons. Bullwhip = order variance ÷ "
                   "demand variance. Fill rate = retailer demand served from stock. Stockout steps = steps where "
                   "retailer demand was not fully met from stock.")

    with bw:
        st.altair_chart(echelon_lines(frames["bw_echelon"]), use_container_width=True)
        st.caption("Click a legend entry to isolate a system. A rising line means order swings grow upstream.")

    with beh:
        b = frames["behaviour"]
        if b.empty:
            st.info("No agent-behaviour data.")
        else:
            st.markdown("**Real LLM runs, design v5:** 160 decisions per system. The model picks an action and "
                        "code computes the order.")
            st.table(behaviour_table(b))
            st.caption("Instruction following: share of answers valid on the first try. Regime accuracy: agreement "
                       "with the deterministic regime detector. Tokens are per decision.")
            if not frames["action_mix"].empty:
                st.markdown("**Action mix by echelon.** Click a legend entry to highlight one action.")
                st.altair_chart(action_mix_bars(frames["action_mix"]), use_container_width=True)

    with earlier:
        e = frames["earlier"]
        if e.empty:
            st.info("No earlier-design data.")
        else:
            st.markdown("**Designs v3.1–v4.1, shock tuning seeds 100–101.** In these designs the LLM wrote the order "
                        "quantity and S6 negotiated. They hold the only real response-time and negotiation measurements.")
            e = e.assign(latency_s=e["latency_ms"] / 1000)
            st.dataframe(
                e[["system", "prompt_version", "seed", "latency_s", "valid_rate", "fallbacks", "rounds", "messages",
                   "tokens_in", "tokens_out", "usd", "note"]],
                hide_index=True, use_container_width=True, placeholder="—",
                column_config={
                    "system": "System", "prompt_version": "Version",
                    "seed": st.column_config.NumberColumn("Seed", format="%d"),
                    "latency_s": st.column_config.NumberColumn("Response time per call (s)", format="%.1f"),
                    "valid_rate": st.column_config.ProgressColumn("Valid output", format="percent", min_value=0, max_value=1),
                    "fallbacks": st.column_config.NumberColumn("Fallbacks", format="%d"),
                    "rounds": st.column_config.NumberColumn("Negotiation rounds", format="%.2f"),
                    "messages": st.column_config.NumberColumn("Messages", format="%.2f"),
                    "tokens_in": st.column_config.NumberColumn("Tokens in", format="%d"),
                    "tokens_out": st.column_config.NumberColumn("Tokens out", format="%d"),
                    "usd": st.column_config.NumberColumn("USD per run", format="$%.4f"), "note": "Note",
                },
            )
            st.caption("Rounds and messages are per decision and apply only to S6. The v4.1 runs are invalid: every "
                       "LLM call was truncated, so their numbers are not decisions.")

    with base:
        t = frames["baselines_test"]
        if t.empty:
            st.info("No test-seed baseline data.")
        else:
            st.markdown("**Test seeds 0–7, traditional and reference policies.** The LLM systems have not run on these seeds.")
            c1, c2 = st.columns([2, 1], vertical_alignment="bottom")
            bmetric = c1.segmented_control(
                "Metric", ["total_cost", "e2e_bullwhip", "fill_rate", "stockout_steps"], default="total_cost",
                format_func=lambda m: {"total_cost": "Total cost", "e2e_bullwhip": "End-to-end bullwhip",
                                       "fill_rate": "Fill rate", "stockout_steps": "Stockout steps"}[m],
                key="base_metric") or "total_cost"
            scen = c2.pills("Scenarios", ["stable", "seasonal", "shock"], selection_mode="multi",
                            default=["stable", "seasonal", "shock"], key="base_scen") or []
            tv = t[t["scenario"].isin(scen)]
            if not tv.empty:
                st.altair_chart(baselines_chart(tv, bmetric), use_container_width=True)

    with notes:
        st.markdown(
            "- **Not run:** S4 and S5 have no real-LLM run in the current design. The test-seed run (3 scenarios × "
            "seeds 0–7) was stopped before any LLM run finished.\n"
            "- **Not logged:** S3/S6 stockout steps, response time and groundedness were not saved in the seed-102 run.\n"
            "- **Tool choice and parameter accuracy:** not applicable. In the current design the tools are deterministic, "
            "so the model never chooses a tool. Older runs record 1.0 for both, but that value is hard-coded and measures nothing.\n"
            "- **Negotiation:** the current S6 makes one orchestrator call per decision with no rounds. Rounds and "
            "messages from the earlier designs are under *Earlier designs*.\n"
            "- **Sources:** `results/v5_paid_run_seed102_attempt3_effort_low.log`, `results/glm_experiment.duckdb`, "
            "`results/v5_final_runs.csv`, built by `scripts/dashboard_existing_data.py`."
        )
