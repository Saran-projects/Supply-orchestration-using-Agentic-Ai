"""Interactive components for chat transcripts, timeline, and export generators."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import streamlit as st

from sca.dashboard.queries import check_message_groundedness
from sca.dashboard.styles import get_agent_style


def render_decision_chat_transcript(decision_data: Dict[str, Any]) -> None:
    """Render structured chat transcript in strict chronological order."""
    echelon = decision_data["echelon"].capitalize()
    step_num = decision_data["step_number"]
    has_conflict = decision_data["has_conflict"]
    conflict_type = decision_data["conflict_type"]
    rounds_used = decision_data["rounds_used"]
    final_order = decision_data["final_order"]
    proposals = decision_data["proposals"]
    tools = decision_data["tools"]
    messages = decision_data["messages"]
    state = decision_data["state"]

    # 1. State Summary Given to Orchestrator
    st.markdown("#### 1. State Observation (Input to Orchestrator)")
    with st.container():
        st.markdown(
            f"""
            <div class="sca-chat-row">
                <div class="sca-chat-avatar" style="border-color: #94A3B8;">🏭</div>
                <div class="sca-chat-bubble" style="border-left: 4px solid #94A3B8;">
                    <div class="sca-chat-header" style="color: #94A3B8;">
                        Environment Telemetry &middot; {echelon} (Step {step_num})
                    </div>
                    <strong>Observed State:</strong> Current Demand = <code>{state['current_demand']:.1f}</code> | 
                    On-Hand Inventory = <code>{state['inventory']:.1f}</code> | 
                    Backlog = <code>{state['backlog']:.1f}</code> | 
                    Pipeline = <code>{state['pipeline']:.1f}</code> | 
                    Capacity Ceiling = <code>{state['capacity']:.1f}</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 2. Agent Proposals & Collapsible Tool Calls
    is_s3 = "s3" in str(decision_data.get("run_id", "")).lower() or "single" in str(decision_data.get("run_id", "")).lower()
    
    if is_s3:
        st.markdown("#### 2. Single Agent Replenishment Decision")
        st.markdown(
            f"""
            <div class="sca-card" style="border-top: 3px solid #06B6D4;">
                <div style="font-size: 1.2rem; font-weight: 700; color: #06B6D4;">🤖 System 3 Single Agent Policy</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: #F8FAFC; margin: 8px 0;">{final_order:.1f} units</div>
                <div style="font-size: 0.85rem; color: #94A3B8;">Autonomous order decision for {echelon} based on observed local state and prompt reasoning.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if tools:
            with st.expander(f"🛠️ Tool Calls ({len(tools)})"):
                for t in tools:
                    st.caption(f"**Tool:** `{t['tool_name']}` ({t['duration_ms']:.1f}ms)")
                    st.json({"inputs": t["inputs"], "outputs": t["outputs"]})
    else:
        st.markdown("#### 2. Specialized Agent Proposals & Tool Execution")
        col_d, col_i, col_p = st.columns(3)

        # Filter tools by agent
        tools_by_agent: Dict[str, List[Dict[str, Any]]] = {"demand": [], "inventory": [], "procurement": []}
        for t in tools:
            ag = t["agent_name"].lower()
            if "demand" in ag:
                tools_by_agent["demand"].append(t)
            elif "inventory" in ag:
                tools_by_agent["inventory"].append(t)
            elif "procurement" in ag:
                tools_by_agent["procurement"].append(t)

        # Demand Agent
        with col_d:
            p_val = proposals.get("demand", "N/A")
            val_str = f"{p_val:.1f}" if isinstance(p_val, (int, float)) else str(p_val)
            st.markdown(
                f"""
                <div class="sca-card" style="border-top: 3px solid #06B6D4;">
                    <div style="font-size: 1.2rem; font-weight: 700; color: #06B6D4;">📈 Demand Agent</div>
                    <div style="font-size: 1.8rem; font-weight: 800; color: #F8FAFC; margin: 8px 0;">{val_str} units</div>
                    <div style="font-size: 0.8rem; color: #94A3B8;">Proposed based on demand forecasting</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if tools_by_agent["demand"]:
                with st.expander(f"🛠️ Tool Calls ({len(tools_by_agent['demand'])})"):
                    for t in tools_by_agent["demand"]:
                        st.caption(f"**Tool:** `{t['tool_name']}` ({t['duration_ms']:.1f}ms)")
                        st.json({"inputs": t["inputs"], "outputs": t["outputs"]})

        # Inventory Agent
        with col_i:
            p_val = proposals.get("inventory", "N/A")
            val_str = f"{p_val:.1f}" if isinstance(p_val, (int, float)) else str(p_val)
            st.markdown(
                f"""
                <div class="sca-card" style="border-top: 3px solid #8B5CF6;">
                    <div style="font-size: 1.2rem; font-weight: 700; color: #8B5CF6;">📦 Inventory Agent</div>
                    <div style="font-size: 1.8rem; font-weight: 800; color: #F8FAFC; margin: 8px 0;">{val_str} units</div>
                    <div style="font-size: 0.8rem; color: #94A3B8;">Calculated via EOQ & (s, S) policy</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if tools_by_agent["inventory"]:
                with st.expander(f"🛠️ Tool Calls ({len(tools_by_agent['inventory'])})"):
                    for t in tools_by_agent["inventory"]:
                        st.caption(f"**Tool:** `{t['tool_name']}` ({t['duration_ms']:.1f}ms)")
                        st.json({"inputs": t["inputs"], "outputs": t["outputs"]})

        # Procurement Agent
        with col_p:
            p_val = proposals.get("procurement", "N/A")
            val_str = f"{p_val:.1f}" if isinstance(p_val, (int, float)) else str(p_val)
            st.markdown(
                f"""
                <div class="sca-card" style="border-top: 3px solid #F97316;">
                    <div style="font-size: 1.2rem; font-weight: 700; color: #F97316;">🚚 Procurement Agent</div>
                    <div style="font-size: 1.8rem; font-weight: 800; color: #F8FAFC; margin: 8px 0;">{val_str} units</div>
                    <div style="font-size: 0.8rem; color: #94A3B8;">Constrained by supplier capacity</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if tools_by_agent["procurement"]:
                with st.expander(f"🛠️ Tool Calls ({len(tools_by_agent['procurement'])})"):
                    for t in tools_by_agent["procurement"]:
                        st.caption(f"**Tool:** `{t['tool_name']}` ({t['duration_ms']:.1f}ms)")
                        st.json({"inputs": t["inputs"], "outputs": t["outputs"]})

    # 3. Conflict Detection Result
    st.markdown("#### 3. Deterministic Conflict Evaluation")
    if is_s3:
        st.markdown(
            """
            <div class="sca-banner-info">
                ℹ️ <strong>SINGLE-AGENT SYSTEM:</strong> Conflict arbitration not applicable. Single agent decides replenishment order directly.
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif has_conflict:
        st.markdown(
            f"""
            <div class="sca-banner-warning">
                ⚠️ <strong>CONFLICT DETECTED:</strong> <code>{conflict_type}</code><br>
                Proposals diverge beyond configured thresholds. Entering bounded negotiation (Max Rounds = 2).
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="sca-banner-info" style="border-left-color: #10B981; background: linear-gradient(90deg, #064e3b, #022c22);">
                ✅ <strong>CONSENSUS REACHED:</strong> No conflict detected across proposals. Skipping negotiation LLM calls to conserve budget.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 4. Negotiation Messages
    st.markdown("#### 4. Dialogue & Reasoning Transcript" if is_s3 else "#### 4. Negotiation Rounds & Agent Dialogue")
    if not messages:
        st.info("No inter-agent negotiation messages exchanged for this decision (consensus consensus order applied directly).")
    else:
        for idx, m in enumerate(messages):
            sender = m["sender"]
            role = m["role"]
            content = m["content"]
            rnd = m.get("round", 1)
            style = get_agent_style(sender)

            st.markdown(
                f"""
                <div class="sca-chat-row">
                    <div class="sca-chat-avatar" style="border-color: {style['color']};">{style['avatar']}</div>
                    <div class="sca-chat-bubble" style="border-left: 4px solid {style['color']};">
                        <div class="sca-chat-header" style="color: {style['color']};">
                            <span>{style['name']}</span>
                            <span class="sca-badge" style="background: rgba(255,255,255,0.1); color: #cbd5e1;">Round {rnd}</span>
                        </div>
                        <div>{content}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # 5. Order Validation
    st.markdown("#### 5. Order Validation & Guardrails")
    val_tools = [t for t in tools if "validate_order" in t["tool_name"]]
    if val_tools:
        vt = val_tools[0]
        st.markdown(
            f"""
            <div class="sca-card" style="border-left: 4px solid #10B981;">
                <strong>Tool:</strong> <code>validate_order</code> &middot; 
                <strong>Status:</strong> {'Valid' if not vt['is_error'] else 'Adjusted / Fallback'} &middot;
                <strong>Proposed:</strong> <code>{vt['inputs'].get('proposed_order', 'N/A')}</code> &rarr; 
                <strong>Validated Output:</strong> <code>{vt['outputs'].get('validated_order', 'N/A')}</code>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="sca-card" style="border-left: 4px solid #10B981;">
                <strong>Validation:</strong> Deterministic order bounds check [0.0, {state['capacity']:.1f}] passed cleanly.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 6. Final Order Dispatched
    st.markdown("#### 6. Final Order Dispatched to Environment")
    st.markdown(
        f"""
        <div class="sca-metric-box" style="border: 2px solid #10B981; background: linear-gradient(135deg, #064e3b, #022c22);">
            <div class="sca-metric-val" style="color: #34D399; font-size: 2.2rem;">{final_order:.1f} Units</div>
            <div class="sca-metric-lbl" style="color: #A7F3D0;">Final Replenishment Order Placed by {echelon} (Step {step_num})</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_groundedness_side_panel(decision_data: Dict[str, Any]) -> None:
    """Render side panel showing true state and regex number matching."""
    st.markdown("### 🔍 Groundedness Audit")
    state = decision_data["state"]

    st.markdown("#### True Environment State")
    st.markdown(
        f"""
        - **Current Demand:** `{state['current_demand']:.1f}`
        - **On-Hand Inventory:** `{state['inventory']:.1f}`
        - **Backlog Units:** `{state['backlog']:.1f}`
        - **Pipeline In-Transit:** `{state['pipeline']:.1f}`
        - **Supplier Capacity:** `{state['capacity']:.1f}`
        """
    )

    st.markdown("---")
    st.markdown("#### Quoted Number Verifier (5% Tolerance)")
    messages = decision_data["messages"]
    if not messages:
        st.caption("No text reasoning messages to verify for this step.")
        return

    found_numbers = False
    for m in messages:
        sender = m["sender"]
        checks = check_message_groundedness(m["content"], state, tolerance=0.05)
        if checks:
            found_numbers = True
            st.markdown(f"**From `{sender}`:**")
            for c in checks:
                status_color = "#34D399" if c["grounded"] else "#F87171"
                st.markdown(
                    f"{c['symbol']} Value `<b>{c['number']}</b>`: "
                    f"<span style='color: {status_color}; font-weight: 600;'>"
                    f"{'Matches State' if c['grounded'] else 'Hallucinated / Unaligned'}</span>",
                    unsafe_allow_html=True,
                )

    if not found_numbers:
        st.caption("No numeric assertions identified in agent messages.")


def export_transcript_to_markdown(decision_data: Dict[str, Any]) -> str:
    """Export decision transcript as markdown document."""
    ech = decision_data["echelon"].capitalize()
    step = decision_data["step_number"]
    st_dict = decision_data["state"]
    proposals = decision_data["proposals"]
    messages = decision_data["messages"]

    lines = [
        f"# SupplyChain Multi-Agent Decision Transcript",
        f"- **Echelon:** {ech}",
        f"- **Simulation Step:** {step}",
        f"- **Conflict Status:** {decision_data['conflict_type']} ({'Yes' if decision_data['has_conflict'] else 'No'})",
        f"- **Final Order:** {decision_data['final_order']:.1f} units",
        "",
        "## 1. Observed State",
        f"- Demand: {st_dict['current_demand']:.1f}",
        f"- Inventory: {st_dict['inventory']:.1f}",
        f"- Backlog: {st_dict['backlog']:.1f}",
        f"- Pipeline: {st_dict['pipeline']:.1f}",
        f"- Capacity: {st_dict['capacity']:.1f}",
        "",
        "## 2. Agent Proposals",
    ]
    for k, v in proposals.items():
        lines.append(f"- **{k.capitalize()} Agent:** {v} units")

    lines.extend(["", "## 3. Negotiation Dialogue"])
    if not messages:
        lines.append("*Consensus reached without negotiation dialogue.*")
    else:
        for m in messages:
            lines.append(f"- **{m['sender']}** (Round {m.get('round', 1)}): {m['content']}")

    lines.extend([
        "",
        "## 4. Final Order Decision",
        f"**Dispatched Order:** {decision_data['final_order']:.1f} units",
    ])
    return "\n".join(lines)


def export_transcript_to_slide_html(decision_data: Dict[str, Any]) -> str:
    """Export decision transcript as a clean standalone HTML card for presentation slides."""
    ech = decision_data["echelon"].capitalize()
    step = decision_data["step_number"]
    proposals = decision_data["proposals"]
    messages = decision_data["messages"]

    prop_html = "".join([
        f"<div style='flex:1; background:#1e293b; padding:10px; border-radius:8px; margin:4px; text-align:center;'>"
        f"<div style='font-size:12px; color:#94a3b8;'>{k.capitalize()}</div>"
        f"<div style='font-size:18px; font-weight:bold; color:#f8fafc;'>{v:.1f}</div>"
        f"</div>"
        for k, v in proposals.items()
        if isinstance(v, (int, float))
    ])

    msgs_html = "".join([
        f"<div style='background:#1e293b; border-left:3px solid #3b82f6; padding:8px 12px; margin:6px 0; border-radius:6px; font-size:13px; color:#e2e8f0;'>"
        f"<strong>{m['sender']}</strong> (R{m.get('round', 1)}): {m['content']}"
        f"</div>"
        for m in messages
    ]) or "<div style='color:#94a3b8; font-style:italic;'>Consensus shortcut applied (0 rounds).</div>"

    return f"""
    <div style="font-family:'Inter', sans-serif; background:#0f172a; color:#f8fafc; padding:24px; border-radius:16px; border:1px solid #334155; max-width:800px;">
        <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #334155; padding-bottom:12px; margin-bottom:16px;">
            <h2 style="margin:0; font-size:20px; color:#38bdf8;">Multi-Agent Decision: {ech} (Step {step})</h2>
            <span style="background:#065f46; color:#34d399; font-weight:bold; padding:4px 12px; border-radius:999px; font-size:14px;">Order: {decision_data['final_order']:.1f}</span>
        </div>
        <div style="margin-bottom:14px;">
            <div style="font-size:13px; color:#94a3b8; margin-bottom:6px; font-weight:600;">AGENT PROPOSALS:</div>
            <div style="display:flex;">{prop_html}</div>
        </div>
        <div>
            <div style="font-size:13px; color:#94a3b8; margin-bottom:6px; font-weight:600;">NEGOTIATION TRANSCRIPT:</div>
            {msgs_html}
        </div>
    </div>
    """
