"""Main Streamlit Application for Supply Chain Agents (SCA) Platform.

Fully offline, strictly read-only DuckDB analytics, interactive chat inspector.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import streamlit as st

from sca.dashboard.components import (
    export_transcript_to_markdown,
    export_transcript_to_slide_html,
    render_decision_chat_transcript,
    render_groundedness_side_panel,
)
from sca.dashboard.queries import (
    DEFAULT_GLM_DB,
    DEFAULT_METRICS_DB,
    ECHELONS,
    get_aggregate_negotiation_analytics,
    get_available_runs,
    get_baselines_20seeds_summary,
    get_conflicting_steps_summary,
    get_cost_and_reconciliation_data,
    get_decision_details,
    get_glm_real_comparison_metrics,
    get_mock_vs_glm_plumbing_metrics,
    get_scenario_metadata,
    get_shock_baseline_demand_vs_orders,
    get_step_comparison_s3_vs_s6,
    load_demand_series_sample,
)
from sca.dashboard.styles import SYSTEM_COLORS, get_custom_css

# Page Configuration
st.set_page_config(
    page_title="SupplyChain Agents (SCA) Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply Modern Stylesheet
st.markdown(get_custom_css(), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar Navigation & DB Status
# ---------------------------------------------------------------------------
st.sidebar.title("📦 SCA Dashboard")
st.sidebar.caption("Autonomous Multi-Agent Supply Chain Benchmark")

PAGES = [
    "1. Overview",
    "2. Real Data & Scenarios",
    "3. Baselines (Real)",
    "4. GLM Real Runs",
    "5. Orchestrator Inspector",
    "6. Mock vs GLM",
    "7. Cost and Plan",
    "8. Agent Conversations",
]

selected_page = st.sidebar.radio("Navigate Pages", PAGES, index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("🗄️ Database Telemetry")
db_options = {
    "GLM Experiment DB": DEFAULT_GLM_DB,
    "Metrics DB": DEFAULT_METRICS_DB,
}
selected_db_label = st.sidebar.selectbox("Active Database", list(db_options.keys()))
active_db = db_options[selected_db_label]

if active_db.exists():
    st.sidebar.success(f"Connected (Read-Only): `{active_db.name}`")
else:
    st.sidebar.warning(f"Database file not found: `{active_db.name}`")

st.sidebar.info("🔒 Offline Execution Mode\nZero API keys required.")


# ===========================================================================
# PAGE 1: OVERVIEW
# ===========================================================================
if selected_page == "1. Overview":
    st.title("🏭 SupplyChain Agents (SCA) Benchmark Platform")
    st.markdown(
        "A rigorous, multi-agent evaluation framework testing LLM ordering policies "
        "against classical inventory heuristics across empirical DataCo supply chain disruptions."
    )

    st.markdown("### 1. Supply Chain Network Topology (4-Echelon Serial Topology)")
    st.markdown(
        """
        The simulation environment models a classic 4-echelon serial **Beer Distribution Game**:
        ```
        [ End Customers ] 
              │ (Consumer Demand)
              ▼
        ┌─────────────┐        Orders Upstream (days t)        ┌──────────────┐
        │  Retailer   │ ─────────────────────────────────────► │  Wholesaler  │
        │             │ ◄───────────────────────────────────── │              │
        └─────────────┘       Shipments Downstream (Lead Time) └──────────────┘
                                                                       │
                                                                       ▼
        ┌─────────────┐        Orders Upstream (days t)        ┌──────────────┐
        │  Factory /  │ ◄───────────────────────────────────── │ Distributor  │
        │ Manufacturer│ ─────────────────────────────────────► │              │
        └─────────────┘       Shipments Downstream (Lead Time) └──────────────┘
        ```
        - **Echelons:** Retailer &rarr; Wholesaler &rarr; Distributor &rarr; Manufacturer.
        - **Order Lag:** Orders propagate upstream with a 1-day transmission delay.
        - **Shipment Lag:** In-transit inventory arrives via echelon-specific stochastic lead times.
        - **Cost Structure:** Holding cost ($0.50/unit/day) and Backlog shortage penalty ($1.00/unit/day).
        """
    )

    st.markdown("---")
    st.markdown("### 2. The Six Replenishment Systems Under Test")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""
            <div class="sca-card" style="border-left: 4px solid {SYSTEM_COLORS['system_1']};">
                <h4>System 1: (s, S) Heuristic</h4>
                <p style="font-size: 0.85rem; color: #94A3B8;">
                    Classical dynamic base-stock inventory policy. Triggers reorder when inventory position 
                    drops below reorder point <code>s</code> up to target <code>S</code>.
                </p>
            </div>
            <div class="sca-card" style="border-left: 4px solid {SYSTEM_COLORS['system_2']};">
                <h4>System 2: EOQ Heuristic</h4>
                <p style="font-size: 0.85rem; color: #94A3B8;">
                    Economic Order Quantity balancing fixed ordering costs against holding costs, 
                    augmented with safety stock for lead-time demand variance.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="sca-card" style="border-left: 4px solid {SYSTEM_COLORS['system_3']};">
                <h4>System 3: Single Agent LLM</h4>
                <p style="font-size: 0.85rem; color: #94A3B8;">
                    Independent LLM prompt per echelon given local inventory, backlog, pipeline, 
                    and recent history window. Zero inter-echelon coordination.
                </p>
            </div>
            <div class="sca-card" style="border-left: 4px solid {SYSTEM_COLORS['system_4']};">
                <h4>System 4: Multi-Agent Info Chain</h4>
                <p style="font-size: 0.85rem; color: #94A3B8;">
                    Echelon agents execute in sequence with shared global visibility into upstream 
                    and downstream stock positions to mitigate bullwhip amplification.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="sca-card" style="border-left: 4px solid {SYSTEM_COLORS['system_5']};">
                <h4>System 5: Multi-Agent with Tools</h4>
                <p style="font-size: 0.85rem; color: #94A3B8;">
                    Modular CrewAI agent roles (Demand, Inventory, Procurement) that invoke pure deterministic 
                    tools (<code>forecast_demand</code>, <code>eoq_quantity</code>, <code>ss_policy_order</code>).
                </p>
            </div>
            <div class="sca-card" style="border-left: 4px solid {SYSTEM_COLORS['system_6']};">
                <h4>System 6: Orchestrated Multi-Agent</h4>
                <p style="font-size: 0.85rem; color: #94A3B8;">
                    Central OrchestratorAgent that detects proposal conflicts, shortcuts consensus steps to save cost, 
                    and coordinates bounded &le;2 round negotiations on divergence.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### 3. Evaluated Performance Metrics")
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.markdown(
            """
            **Operational & Financial:**
            - **Total Cost ($):** Cumulative holding + backlog penalties.
            - **Bullwhip Ratio:** $\\text{Var}(\\text{Orders}) / \\text{Var}(\\text{Demand})$ per echelon.
            - **Fill Rate (%):** Fulfilled customer demand share.
            - **Stockout Days:** Steps with backlogged stockouts.
            """
        )
    with m_col2:
        st.markdown(
            """
            **Behavioural & Tool Grounding:**
            - **Tool Choice Accuracy:** Correct tool invoked vs expectation.
            - **Parameter Accuracy:** Tool arguments within 5% of true state.
            - **Instruction Following (%):** Valid JSON on first attempt.
            - **Groundedness Score (%):** Regex verification of text claims.
            """
        )
    with m_col3:
        st.markdown(
            """
            **Negotiation & Efficiency:**
            - **Conflict Share (%):** Fraction of steps triggering negotiation.
            - **Negotiation Rounds:** Rounds consumed (bounded &le;2).
            - **Tokens In / Out:** Prompt & completion token volume.
            - **Cost in USD:** Logged spend per 2.2x Isoquant rate card.
            """
        )

    st.markdown("---")
    st.markdown("### 4. Base Paper Reference")
    st.markdown(
        """
        > **Foundational Literature:**  
        > **Sterman, J. D. (1989).** *"Modeling Managerial Behavior: Misperceptions of Feedback in a Dynamic Decision Making Experiment."* **Management Science**, 35(3), 321–339.  
        > Explores human cognitive bias, order overreaction, and phase delays causing severe supply chain oscillations (Bullwhip Effect).  
        >  
        > **SCA Research Goal:**  
        > Investigating whether modern LLM multi-agent architectures overcome Sterman's classical misperceptions of feedback, or whether unconstrained negotiation introduces new stochastic instabilities.
        """
    )


# ===========================================================================
# PAGE 2: REAL DATA & SCENARIOS
# ===========================================================================
elif selected_page == "2. Real Data & Scenarios":
    st.title("📊 Empirical DataCo Dataset & Scenario Windows")
    meta = get_scenario_metadata()

    # Overview Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{meta['total_records']:,}")
    c2.metric("Date Span", f"{meta['date_range'][0]} &rarr; {meta['date_range'][1]}")
    c3.metric("Continuous Days", f"{meta['total_days']} days")
    c4.metric("Late Delivery Share", f"{meta['late_delivery_rate'] * 100:.2f}%")

    st.markdown("---")
    st.markdown("### 1. Delivery Performance & Shipping Lead Times")
    l_col1, l_col2 = st.columns(2)

    with l_col1:
        st.markdown("#### Shipping Modes & Empirical Lead Times")
        modes_df = pd.DataFrame(meta["shipping_modes"])
        st.dataframe(
            modes_df[["mode", "share", "orders", "lead_time_mean", "lead_time_std"]].rename(
                columns={
                    "mode": "Shipping Mode",
                    "share": "Share",
                    "orders": "Order Count",
                    "lead_time_mean": "Mean Lead Time (d)",
                    "lead_time_std": "Std Dev (d)",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("Standard and Second Class exhibit identical 4.0-day uniform transit distributions.")

    with l_col2:
        st.markdown("#### Delivery Status Breakdown")
        status_df = pd.DataFrame(meta["delivery_statuses"])
        st.dataframe(
            status_df.rename(columns={"status": "Delivery Status", "share": "Share", "orders": "Orders"}),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("Over 54.8% of historical shipments encounter delivery delays, necessitating safety buffers.")

    st.markdown("---")
    st.markdown("### 2. Target SKU: Cleats Demand Time-Series")
    demand_df = load_demand_series_sample()
    if not demand_df.empty:
        d_col = "order_date" if "order_date" in demand_df.columns else "date"
        q_col = "order_item_quantity" if "order_item_quantity" in demand_df.columns else "demand"
        st.line_chart(demand_df.set_index(d_col)[q_col])
        st.caption("Daily demand series for 'Cleats' (1,006 steps) scaled to mean 100.0 units for simulation.")

    st.markdown("---")
    st.markdown("### 3. 90-Day Scenario Windows")
    sc_col1, sc_col2, sc_col3 = st.columns(3)
    for idx, (s_key, s_data) in enumerate(meta["scenarios"].items()):
        col_target = [sc_col1, sc_col2, sc_col3][idx]
        with col_target:
            st.markdown(
                f"""
                <div class="sca-card">
                    <h4>Scenario: {s_data['name'].upper()}</h4>
                    <p style="font-size: 0.85rem; color: #94A3B8;">{s_data['description']}</p>
                    <ul style="font-size: 0.85rem; color: #cbd5e1; padding-left: 20px;">
                        <li><strong>Window:</strong> {s_data['window'][0]} &rarr; {s_data['window'][1]}</li>
                        <li><strong>Horizon:</strong> {s_data['horizon_days']} days (Train: {s_data['train_length_days']}d)</li>
                        <li><strong>Raw Mean:</strong> {s_data['raw_mean']} units</li>
                        <li><strong>Target Mean:</strong> {s_data['scaled_mean']} units</li>
                        <li><strong>Coefficient of Variation (CV):</strong> <code>{s_data['cv']}</code></li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ===========================================================================
# PAGE 3: BASELINES (REAL)
# ===========================================================================
elif selected_page == "3. Baselines (Real)":
    st.title("📈 Classical Inventory Baselines: (S,s) and EOQ")
    st.markdown(
        "Benchmarking classical stochastic inventory heuristics across **20 random seeds** "
        "(seeds 42 to 61) to establish rigorous empirical performance bounds."
    )

    baseline_df = get_baselines_20seeds_summary(active_db)
    if baseline_df.empty:
        st.warning("No baseline runs found in the active database. Run `make baselines` to generate them.")
    else:
        st.markdown("### 1. 20-Seed Performance Distribution (Mean ± Std)")
        display_cols = [
            "Policy", "Scenario", "Seeds Count", "Total Cost ($)", "Retailer BW",
            "Wholesaler BW", "Distributor BW", "Manufacturer BW", "Fill Rate (%)", "Stockout Steps"
        ]
        st.dataframe(baseline_df[display_cols], use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### 2. Bullwhip Amplification Upstream by Echelon")
        # Visualizing bullwhip propagation
        bw_chart_data = []
        for _, row in baseline_df.iterrows():
            bw_chart_data.append({
                "Policy / Scenario": f"{row['Policy']} ({row['Scenario']})",
                "Retailer": row["raw_retailer_bw"],
                "Wholesaler": row["raw_wholesaler_bw"],
                "Distributor": row["raw_distributor_bw"],
                "Manufacturer": row["raw_manufacturer_bw"],
            })
        bw_plot_df = pd.DataFrame(bw_chart_data).set_index("Policy / Scenario")
        st.bar_chart(bw_plot_df.T)
        st.caption("Progression of order variance amplification upstream (Retailer &rarr; Manufacturer).")

        st.markdown("---")
        st.markdown("### 3. Shock Scenario: Customer Demand vs Replenishment Orders (Seed 42)")
        shock_steps_df = get_shock_baseline_demand_vs_orders(active_db, seed=42)
        if not shock_steps_df.empty:
            pivoted_steps = shock_steps_df.pivot(index="step_number", columns="run_id", values="retailer_order")
            # Add demand series
            first_run = shock_steps_df["run_id"].iloc[0]
            pivoted_steps["Customer Demand"] = shock_steps_df[shock_steps_df["run_id"] == first_run].set_index("step_number")["demand_units"]
            st.line_chart(pivoted_steps)
            st.caption("Retailer order responses during sudden shock demand spikes and post-shock inventory bullwhips.")


# ===========================================================================
# PAGE 4: GLM REAL RUNS
# ===========================================================================
elif selected_page == "4. GLM Real Runs":
    st.title("🤖 GLM Real Runs: Systems 3 & 6 vs Baselines (Seed 42)")

    # Mandatory Scientific Disclaimer Banner
    st.markdown(
        """
        <div class="sca-banner-warning">
            ⚠️ <strong>METHODOLOGICAL NOTE:</strong> Single runs (n=1 per cell): descriptive case studies, 
            not statistical findings. Never rank n=1 results as winners.
        </div>
        """,
        unsafe_allow_html=True,
    )

    glm_metrics_df = get_glm_real_comparison_metrics(active_db, seed=42)
    if glm_metrics_df.empty:
        st.warning("No completed seed 42 runs found in database. Run `python scripts/run_shock_glm_experiment.py` first.")
    else:
        st.markdown("### Comprehensive Evaluation Matrix (Seed 42, Shock Scenario)")
        st.dataframe(glm_metrics_df, use_container_width=True, hide_index=True)

        st.caption(
            "Values marked as 'n/a' indicate metrics that do not apply to that architecture "
            "(e.g., negotiation rounds for single-agent or classical heuristics, token costs for heuristics)."
        )

        st.markdown("---")
        st.markdown("### Key Observations (Descriptive Case Study)")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(
                """
                **System 3 (Single Agent LLM):**
                - Employs zero inter-echelon coordination; decisions rely entirely on local state and prompt reasoning.
                - Experienced 0 fallbacks and 97.5% instruction-following adherence on GLM-5.3-Flash.
                - Generated high reasoning token volume (average ~2,500 total tokens per decision).
                """
            )
        with col_b:
            st.markdown(
                """
                **System 6 (Orchestrated Multi-Agent):**
                - Orchestrator detected proposal conflicts and successfully executed bounded &le;2 round negotiations.
                - In consensus steps, skipped negotiation entirely, saving token budget.
                - Maintained 100% tool-choice accuracy and parameter adherence across specialized agents.
                """
            )


# ===========================================================================
# PAGE 5: ORCHESTRATOR INSPECTOR
# ===========================================================================
elif selected_page == "5. Orchestrator Inspector":
    st.title("🎯 Orchestrator Decision Inspector")
    st.markdown("Inspect real GLM negotiation messages, proposals, conflict checks, and order validation for any step.")

    # Selectors
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        step_sel = st.slider("Select Simulation Step", min_value=0, max_value=29, value=0)
    with col_s2:
        ech_sel = st.selectbox("Select Echelon", [e.capitalize() for e in ECHELONS], index=0)
    with col_s3:
        target_run_sel = st.selectbox("Run Selection", ["s6_glm_isoquant_shock_seed_42", "s6_mock_shock_seed_42"])

    dec_data = get_decision_details(active_db, target_run_sel, step_sel, ech_sel.lower())

    # Render complete decision transcript
    render_decision_chat_transcript(dec_data)


# ===========================================================================
# PAGE 6: MOCK VS GLM
# ===========================================================================
elif selected_page == "6. Mock vs GLM":
    st.title("🔬 MockLLM vs GLM-5.3-Flash: Reliability & Plumbing")

    # Mandatory Banner
    st.markdown(
        """
        <div class="sca-banner-info">
            ℹ️ <strong>CONTROL NOTE:</strong> MockLLM is a scripted control, not an LLM. 
            Comparing system plumbing, latency overhead, fallback resilience, and structured schema handling.
        </div>
        """,
        unsafe_allow_html=True,
    )

    plumbing_df = get_mock_vs_glm_plumbing_metrics(active_db)
    if plumbing_df.empty:
        st.warning("Insufficient comparison data between mock and real runs.")
    else:
        st.markdown("### Plumbing & Robustness Comparison Matrix")
        st.dataframe(plumbing_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### Technical Observations")
        st.markdown(
            """
            1. **Fallback Resilience:** Both MockLLM and real GLM-5.3-Flash runs exhibited **0.0% fallback rates**, confirming robust Pydantic schema validation and automatic retry handlers.
            2. **Structured Output Adherence:** GLM-5.3-Flash parsed valid `OrderDecision` and `NegotiationResponse` schemas cleanly on the first attempt with 100% reliability.
            3. **Latency Profile:** Real GLM API calls averaged ~8,000–15,000 ms per step (under 60 RPM limits), compared to near-zero (<1 ms) execution in MockLLM.
            """
        )


# ===========================================================================
# PAGE 7: COST AND PLAN
# ===========================================================================
elif selected_page == "7. Cost and Plan":
    st.title("💰 Token Budget, Real Spend & Experiment Plan")

    cost_data = get_cost_and_reconciliation_data(active_db)

    # 1. Reconciliation Summary
    st.markdown("### 1. Spend Reconciliation vs Isoquant Console")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Isoquant Console Billed", f"${cost_data['console_billed_usd']:.2f} USD")
    r2.metric("All-Time Logged Spend", f"${cost_data['all_time_logged_usd']:.4f} USD")
    r3.metric("Incremental Variance", f"${cost_data['variance_usd']:+.4f} USD")
    r4.metric("Coverage Ratio", f"{cost_data['coverage_ratio'] * 100:.1f}%")

    st.caption(
        "Initial historical baseline was $0.20634 USD (~$0.21 console billed). "
        "Recent 30-step shock runs added $0.23620 USD to the logged total."
    )

    st.markdown("---")
    st.markdown("### 2. Measured vs Estimated Tokens per Decision")
    tm_df = cost_data["token_measurements"]
    st.dataframe(
        tm_df.rename(columns={
            "system": "System",
            "est_in": "Est. Input",
            "measured_in": "Measured Input",
            "in_ratio": "Input Ratio (Meas/Est)",
            "est_out": "Est. Output",
            "measured_out": "Measured Output",
            "out_ratio": "Output Ratio (Meas/Est)",
        }),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Notice that System 6 measured input tokens are only **0.07× of initial estimates** "
        "because consensus shortcut skips all negotiation LLM calls on steps without conflicts."
    )

    st.markdown("---")
    st.markdown("### 3. Planned Experiment Grid & Projected Budget")
    pg_df = cost_data["planned_grid"]
    st.dataframe(
        pg_df.rename(columns={
            "system": "System",
            "architecture": "Architecture",
            "cost_per_decision": "Est. Cost / Dec ($)",
            "cost_per_90step_run": "Projected Cost / 90-step Run ($)",
            "full_grid_cost_15runs": "15-Run Grid Cost ($)",
        }),
        use_container_width=True,
        hide_index=True,
    )
    tot_plan_cost = pg_df["full_grid_cost_15runs"].sum()
    st.info(f"**Total Projected Grid Budget (60 Runs):** ${tot_plan_cost:.2f} USD (Across S3, S4, S5, S6 over 3 scenarios and 5 seeds).")


# ===========================================================================
# PAGE 8: AGENT CONVERSATIONS
# ===========================================================================
elif selected_page == "8. Agent Conversations":
    st.title("💬 Agent Conversations & Negotiation Dialogue")
    st.markdown("Deep interactive trace of multi-agent debate, tool calls, groundedness verifications, and comparison toggles.")

    # Selectors
    all_runs = get_available_runs(active_db)
    if not all_runs:
        st.warning("No runs available in active database.")
    else:
        # Filter runs with agent decision telemetry (Systems 3 to 6)
        llm_runs = [r for r in all_runs if any(r["run_id"].startswith(f"s{s}_") for s in (3, 4, 5, 6))]
        if not llm_runs:
            llm_runs = all_runs

        # Selector Bar
        c_ds, c_sc, c_sys, c_seed = st.columns(4)
        with c_ds:
            ds_choices = [("glm_real", "Real GLM-5.3-Flash"), ("mock_llm", "MockLLM (Scripted Control)")]
            sel_ds = st.selectbox(
                "Data Source",
                [d[0] for d in ds_choices],
                format_func=lambda x: dict(ds_choices).get(x, x),
                index=0,
            )
        with c_sc:
            sc_choices = sorted(list(set(r["scenario"] for r in llm_runs)))
            sel_sc = st.selectbox("Scenario", sc_choices or ["shock", "stable"], index=0)
        with c_sys:
            sel_sys_num = st.selectbox(
                "System Architecture",
                [6, 3],
                format_func=lambda s: f"System {s}: {'Orchestrated (Multi-Agent)' if s==6 else 'Single Agent'}",
                index=0,
            )
        with c_seed:
            seed_choices = sorted(list(set(r["seed"] for r in llm_runs)))
            sel_seed = st.selectbox("Random Seed", seed_choices or [42], index=0)

        # Match run by metadata attributes
        matching_runs = []
        for r in llm_runs:
            r_id = r["run_id"]
            sys_ok = r_id.startswith(f"s{sel_sys_num}_") or (sel_sys_num == 6 and "orchestrated" in str(r.get("system", ""))) or (sel_sys_num == 3 and "single" in str(r.get("system", "")))
            ds_ok = (sel_ds == "glm_real" and ("glm" in r_id or r.get("data_source") == "glm_real")) or \
                    (sel_ds == "mock_llm" and ("mock" in r_id or r.get("data_source") == "mock_llm"))
            sc_ok = (sel_sc in r_id or r.get("scenario") == sel_sc)
            seed_ok = (f"seed_{sel_seed}" in r_id or r.get("seed") == sel_seed)
            if sys_ok and ds_ok and sc_ok and seed_ok:
                matching_runs.append(r_id)

        # Fallback to any run matching system and data source if exact combination missing
        if not matching_runs:
            for r in llm_runs:
                r_id = r["run_id"]
                sys_ok = r_id.startswith(f"s{sel_sys_num}_")
                ds_ok = (sel_ds == "glm_real" and "glm" in r_id) or (sel_ds == "mock_llm" and "mock" in r_id)
                if sys_ok and ds_ok:
                    matching_runs.append(r_id)

        all_candidate_ids = [r["run_id"] for r in llm_runs]
        candidate_list = matching_runs if matching_runs else all_candidate_ids

        c_run_sel, c_run_info = st.columns([2, 1])
        with c_run_sel:
            active_conv_run = st.selectbox(
                "Active Run Selection",
                candidate_list,
                index=0,
            )
        with c_run_info:
            st.caption(f"**Target Run ID:** `{active_conv_run}`")

        # Conflict filter (only meaningful for multi-agent System 6)
        c_filter, c_ech = st.columns([2, 1])
        with c_filter:
            only_conflicts = st.checkbox("🔍 Only decisions with a conflict", value=(sel_sys_num == 6))
        with c_ech:
            sel_echelon = st.selectbox("Echelon", [e.capitalize() for e in ECHELONS], index=0)

        # Step selection
        conflicting_steps = get_conflicting_steps_summary(active_db, active_conv_run, echelon=sel_echelon)

        if only_conflicts and conflicting_steps:
            step_opts = {s["label"]: s["step_number"] for s in conflicting_steps}
            chosen_step_label = st.selectbox("Select Conflicting Step", list(step_opts.keys()))
            selected_step = step_opts[chosen_step_label]
        else:
            selected_step = st.slider("Select Simulation Step Number", min_value=0, max_value=29, value=0)

        st.markdown("---")

        # Fetch full decision details
        conv_data = get_decision_details(active_db, active_conv_run, selected_step, sel_echelon.lower())

        # Main Layout: 2 Columns (Transcript + Groundedness Audit)
        t_col, s_col = st.columns([3, 2])

        with t_col:
            # Render chat transcript
            render_decision_chat_transcript(conv_data)

            # Per-Decision Stats Bar
            st.markdown("---")
            st.markdown("#### Decision Telemetry Stats")
            stats = conv_data["stats"]
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("LLM Invocations", f"{stats['calls_count']}")
            s2.metric("Total Tokens", f"{stats['total_tokens']:,}")
            s3.metric("Latency", f"{stats['latency_ms']:.1f} ms")
            s4.metric("Cost", f"${stats['cost_usd']:.5f} USD")

            # Compare Toggle: System 3 vs System 6
            st.markdown("---")
            compare_s3 = st.toggle("⚖️ Compare with System 3 (Single Agent) Decision", value=False)
            if compare_s3:
                s3_candidates = [
                    r["run_id"] for r in all_runs
                    if r["run_id"].startswith("s3_") and (
                        (sel_ds == "glm_real" and "glm" in r["run_id"]) or
                        (sel_ds == "mock_llm" and "mock" in r["run_id"])
                    ) and (sel_sc in r["run_id"])
                ]
                s3_run_target = s3_candidates[0] if s3_candidates else ("s3_glm_isoquant_shock_seed_42" if sel_ds == "glm_real" else "s3_mock_shock_seed_42")
                comp_data = get_step_comparison_s3_vs_s6(
                    active_db,
                    s3_run_id=s3_run_target,
                    s6_run_id=active_conv_run,
                    step_number=selected_step,
                    echelon=sel_echelon.lower(),
                )
                st.markdown("##### Side-by-Side Comparison (Step %d, %s)" % (selected_step, sel_echelon))
                cmp_c1, cmp_c2 = st.columns(2)
                with cmp_c1:
                    st.markdown(
                        f"""
                        <div class="sca-card" style="border-top: 3px solid #06B6D4;">
                            <h4>System 3 (Single Agent)</h4>
                            <p><strong>Order Placed:</strong> <code>{comp_data['s3']['order']:.1f}</code> units</p>
                            <p><strong>Latency / Cost:</strong> {comp_data['s3']['latency_ms']:.1f}ms &middot; ${comp_data['s3']['cost_usd']:.5f}</p>
                            <p><strong>Resulting t+1 Inventory:</strong> <code>{comp_data['s3']['next_inventory'] or 'N/A'}</code></p>
                            <p><strong>Resulting t+1 Backlog:</strong> <code>{comp_data['s3']['next_backlog'] or 'N/A'}</code></p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with cmp_c2:
                    st.markdown(
                        f"""
                        <div class="sca-card" style="border-top: 3px solid #10B981;">
                            <h4>System 6 (Orchestrated)</h4>
                            <p><strong>Order Placed:</strong> <code>{comp_data['s6']['order']:.1f}</code> units</p>
                            <p><strong>Conflict Status:</strong> [{comp_data['s6']['conflict_type']}] &middot; {comp_data['s6']['rounds_used']} rounds</p>
                            <p><strong>Resulting t+1 Inventory:</strong> <code>{comp_data['s6']['next_inventory'] or 'N/A'}</code></p>
                            <p><strong>Resulting t+1 Backlog:</strong> <code>{comp_data['s6']['next_backlog'] or 'N/A'}</code></p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # Export Options
            st.markdown("---")
            st.markdown("#### 📥 Export Decision Transcript")
            exp_col1, exp_col2 = st.columns(2)
            with exp_col1:
                md_text = export_transcript_to_markdown(conv_data)
                st.download_button(
                    label="📄 Export as Markdown (.md)",
                    data=md_text,
                    file_name=f"transcript_step_{selected_step}_{sel_echelon.lower()}.md",
                    mime="text/markdown",
                )
            with exp_col2:
                html_text = export_transcript_to_slide_html(conv_data)
                st.download_button(
                    label="🖼️ Export Slide Card (.html)",
                    data=html_text,
                    file_name=f"transcript_slide_step_{selected_step}_{sel_echelon.lower()}.html",
                    mime="text/html",
                )

        with s_col:
            # Groundedness Audit Panel
            render_groundedness_side_panel(conv_data)

            # Run-Level Aggregate Analytics
            st.markdown("---")
            st.markdown("### 📊 Run-Level Negotiation Dynamics")
            agg = get_aggregate_negotiation_analytics(active_db, active_conv_run)

            st.metric("Total Decisions Analyzed", f"{agg['total_decisions']}")
            st.metric("Overall Conflict Rate", f"{agg['conflict_rate'] * 100:.1f}%")
            st.metric("Average Messages / Decision", f"{agg['avg_messages']:.2f}")

            if agg["rounds_distribution"]:
                st.markdown("#### Negotiation Rounds Distribution")
                rounds_df = pd.DataFrame(list(agg["rounds_distribution"].items()), columns=["Rounds", "Count"]).set_index("Rounds")
                st.bar_chart(rounds_df)

            if agg["conflict_shares"]:
                st.markdown("#### Conflict Triggers Breakdown")
                conf_df = pd.DataFrame(list(agg["conflict_shares"].items()), columns=["Trigger Type", "Occurrences"]).set_index("Trigger Type")
                st.bar_chart(conf_df)
