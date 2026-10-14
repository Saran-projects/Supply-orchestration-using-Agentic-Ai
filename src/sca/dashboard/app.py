"""Main Streamlit Application for Supply Chain Agents (SCA) Platform.

Fully offline, strictly read-only DuckDB analytics, interactive chat inspector.
"""

from __future__ import annotations

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
    get_conflicting_steps_summary,
    get_decision_details,
    get_step_comparison_s3_vs_s6,
)
from sca.dashboard.results import render_results_page
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
    "Results: Systems 1–6",
    "Overview",
    "Orchestrator Inspector",
    "Agent Conversations",
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
# RESULTS: SYSTEMS 1-6 (default page)
# ===========================================================================
if selected_page == "Results: Systems 1–6":
    render_results_page()

# ===========================================================================
# PAGE 1: OVERVIEW
# ===========================================================================
elif selected_page == "Overview":
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
elif selected_page == "Orchestrator Inspector":
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
elif selected_page == "Agent Conversations":
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
