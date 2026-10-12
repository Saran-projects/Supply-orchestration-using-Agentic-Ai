"""Styling tokens, custom CSS, badges, and avatars for the SCA Dashboard."""

from __future__ import annotations

# System Color Palette (Curated HSL/Hex Harmonious Tokens)
SYSTEM_COLORS = {
    "system_1": "#3B82F6",  # Blue
    "baseline_s_S": "#3B82F6",
    "(S,s)": "#3B82F6",
    "system_2": "#6366F1",  # Indigo
    "baseline_eoq": "#6366F1",
    "EOQ": "#6366F1",
    "system_3": "#06B6D4",  # Cyan
    "single_agent": "#06B6D4",
    "system_4": "#F59E0B",  # Amber
    "multi_agent_info": "#F59E0B",
    "system_5": "#8B5CF6",  # Purple
    "multi_agent_tools": "#8B5CF6",
    "system_6": "#10B981",  # Emerald
    "orchestrated": "#10B981",
}

# Agent Personas: Avatar & Accent Color
AGENT_PERSONAS = {
    "orchestrator": {
        "name": "Orchestrator",
        "avatar": "🎯",
        "color": "#10B981",
        "bg_color": "#064e3b",
        "border": "#059669",
    },
    "demand": {
        "name": "Demand Agent",
        "avatar": "📈",
        "color": "#06B6D4",
        "bg_color": "#164e63",
        "border": "#0891b2",
    },
    "inventory": {
        "name": "Inventory Agent",
        "avatar": "📦",
        "color": "#8B5CF6",
        "bg_color": "#4c1d95",
        "border": "#7c3aed",
    },
    "procurement": {
        "name": "Procurement Agent",
        "avatar": "🚚",
        "color": "#F97316",
        "bg_color": "#7c2d12",
        "border": "#ea580c",
    },
    "mediator": {
        "name": "Mediator",
        "avatar": "⚖️",
        "color": "#3B82F6",
        "bg_color": "#1e3a8a",
        "border": "#2563eb",
    },
    "single": {
        "name": "Single Agent (LLM)",
        "avatar": "🤖",
        "color": "#06B6D4",
        "bg_color": "#164e63",
        "border": "#0891b2",
    },
    "system": {
        "name": "Environment State",
        "avatar": "🏭",
        "color": "#94A3B8",
        "bg_color": "#1e293b",
        "border": "#475569",
    },
}


def get_agent_style(agent_name: str) -> dict:
    """Resolve avatar and visual styles for an agent identifier."""
    lowered = agent_name.lower()
    for key, val in AGENT_PERSONAS.items():
        if key in lowered:
            return val
    return AGENT_PERSONAS["system"]


def get_custom_css() -> str:
    """Return polished modern styling for the Streamlit dashboard."""
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Fira+Code:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Sleek card containers */
    .sca-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 16px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    
    .sca-metric-box {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 14px 18px;
        text-align: center;
    }
    
    .sca-metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #F8FAFC;
    }
    
    .sca-metric-lbl {
        font-size: 0.8rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8;
        margin-top: 4px;
    }
    
    /* Scientific banner warning */
    .sca-banner-warning {
        background: linear-gradient(90deg, #78350f, #451a03);
        border-left: 4px solid #f59e0b;
        color: #fef3c7;
        padding: 12px 18px;
        border-radius: 6px;
        font-size: 0.92rem;
        font-weight: 500;
        margin: 12px 0 20px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    
    /* Control note banner */
    .sca-banner-info {
        background: linear-gradient(90deg, #1e3a8a, #0f172a);
        border-left: 4px solid #38bdf8;
        color: #e0f2fe;
        padding: 12px 18px;
        border-radius: 6px;
        font-size: 0.92rem;
        font-weight: 500;
        margin: 12px 0 20px 0;
    }
    
    /* Chat transcript bubbles */
    .sca-chat-row {
        display: flex;
        align-items: flex-start;
        margin-bottom: 14px;
        gap: 12px;
    }
    
    .sca-chat-avatar {
        font-size: 1.5rem;
        width: 38px;
        height: 38px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        background: #1e293b;
        border: 1px solid rgba(255, 255, 255, 0.15);
        flex-shrink: 0;
    }
    
    .sca-chat-bubble {
        flex-grow: 1;
        background: #1e293b;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 12px 16px;
        color: #e2e8f0;
        font-size: 0.92rem;
        line-height: 1.5;
    }
    
    .sca-chat-header {
        font-weight: 600;
        font-size: 0.85rem;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .sca-badge {
        font-size: 0.72rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 9999px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .sca-collapsible-tool {
        background: rgba(15, 23, 42, 0.8);
        border: 1px dashed rgba(255, 255, 255, 0.15);
        border-radius: 8px;
        padding: 10px 14px;
        margin: 8px 0;
        font-family: 'Fira Code', monospace;
        font-size: 0.8rem;
    }
    
    /* Code formatting */
    code {
        font-family: 'Fira Code', monospace;
        font-size: 0.88em;
    }
    </style>
    """
