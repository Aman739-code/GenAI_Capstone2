"""
🌞 Solar Grid Optimization Agent — Streamlit UI
Professional dashboard for the Intelligent Solar Energy Forecasting
& Agentic Grid Optimization system.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import textwrap

# ─── Ensure project root is on path ───
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.graph import run_pipeline
from config.settings import get_api_key
from models.forecast import generate_forecast

# ─── Logging ───
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Page Configuration ───
st.set_page_config(
    page_title="Solar Grid Optimization Agent",
    page_icon="🌞",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Design System & CSS ───
UI_CSS = """
<style>
    /* Premium Font & Global Root */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Outfit', 'Inter', sans-serif;
        background-color: #05070A;
        background-image: 
            radial-gradient(at 0% 0%, rgba(247, 201, 72, 0.05) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(255, 107, 53, 0.05) 0px, transparent 50%);
        color: #E2E8F0;
    }

    /* Glass Effect Utility */
    .glass-card {
        background: rgba(17, 21, 30, 0.6);
        backdrop-filter: blur(25px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
    }

    /* Sidebar Logo */
    .sidebar-logo {
        font-size: 1.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #F7C948 0%, #FF6B35 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }

    /* KPI Cards (Restored Horizontal Look) */
    .kpi-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    .kpi-card:hover {
        border-color: rgba(255, 107, 53, 0.4);
        background: rgba(255, 255, 255, 0.04);
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #FFFFFF;
        line-height: 1.2;
    }
    .kpi-label {
        color: #94A3B8;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 0.25rem;
    }

    /* Modern Tabs Restoration */
    .stTabs [data-baseweb="tab-list"] { gap: 2rem; }
    .stTabs [data-baseweb="tab"] { font-weight: 600; color: #64748B; }
    .stTabs [aria-selected="true"] { color: #FF6B35 !important; }

    /* Terminal/Reasoning */
    .terminal-box {
        font-family: 'JetBrains Mono', monospace;
        background: #020408;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 1rem;
        font-size: 0.85rem;
        color: #CBD5E0;
    }

    /* Custom Headers */
    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 1.25rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Hide redundant Streamlit UI */
    #MainMenu, footer, header, .stDeployButton { visibility: hidden; display: none !important; }

    /* Hide sidebar close/collapse button — sidebar must always stay open */
    [data-testid="collapsedControl"] { display: none !important; }
    button[kind="header"] { display: none !important; }
    [data-testid="stSidebar"] > div:first-child > button { display: none !important; }
    .stSidebar [data-testid="baseButton-header"] { display: none !important; }
    [data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] { display: none !important; }

    /* Section headers */
    .section-header {
        color: #FF6B35;
        font-weight: 600;
        font-size: 1.3rem;
        border-bottom: 2px solid rgba(255, 107, 53, 0.3);
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }

    /* Storage action badges */
    .action-charge { background: linear-gradient(135deg, #4CAF50, #66BB6A); }
    .action-discharge { background: linear-gradient(135deg, #FF6B35, #FF8C42); }
    .action-hold { background: linear-gradient(135deg, #607D8B, #78909C); }

    .action-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 6px;
        color: white;
        font-weight: 600;
        font-size: 0.8rem;
    }

    /* Divider */
    .custom-divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255, 107, 53, 0.3), transparent);
        margin: 1.5rem 0;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
    }
</style>
"""

st.markdown(UI_CSS, unsafe_allow_html=True)

# ─── Helper Functions ───

def render_kpi(value: str, label: str) -> str:
    """Render a compact horizontal KPI card."""
    return f"""
    <div class="kpi-card">
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
    </div>
    """


def render_risk_badge(level: str) -> str:
    """Render a clean risk status label."""
    lvl = level.upper() if level else "MEDIUM"
    colors = {
        "LOW": "#00C853",
        "MEDIUM": "#FFA726",
        "HIGH": "#FF1744",
        "CRITICAL": "#FF5251"
    }
    color = colors.get(lvl, "#FFA726")
    return f'<span style="color: {color};">{lvl}</span>'


def render_node_status(name: str, status: str) -> str:
    """Render a pipeline node status badge."""
    if status == "active":
        return f'<div style="color:#FF6B35; font-weight:700;">● {name}</div>'
    elif status == "complete":
        return f'<div style="color:#00C853;">✓ {name}</div>'
    else:
        return f'<div style="color:#4A5568;">○ {name}</div>'



def create_forecast_chart(forecast_data: dict) -> go.Figure:
    """Create a premium interactive forecast power output chart."""
    predictions = forecast_data.get("hourly_predictions", [])
    features = forecast_data.get("raw_features", {})
    timestamps = features.get("timestamps", list(range(len(predictions))))

    df = pd.DataFrame({
        "Timestamp": pd.to_datetime(timestamps) if timestamps else range(len(predictions)),
        "Power Output (kW)": predictions,
    })

    fig = go.Figure()

    # Main Area Trace
    fig.add_trace(go.Scatter(
        x=df["Timestamp"],
        y=df["Power Output (kW)"],
        fill="tozeroy",
        fillcolor="rgba(255, 107, 53, 0.1)",
        line=dict(color="#FF6B35", width=3, shape="spline"),
        name="Solar Forecast",
        hovertemplate="<b>%{x}</b><br>Generation: %{y:.2f} kW<extra></extra>",
    ))

    # Cloud Cover Trace
    if "cloud_cover" in features:
        fig.add_trace(go.Scatter(
            x=df["Timestamp"],
            y=[c * 100 for c in features["cloud_cover"]],
            line=dict(color="rgba(113, 128, 150, 0.4)", width=1, dash="dot"),
            name="Cloud Density",
            yaxis="y2",
            hovertemplate="Cloud: %{y:.1f}%<extra></extra>",
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=450,
        margin=dict(l=0, r=0, t=20, b=0),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=False),
        yaxis2=dict(overlaying="y", side="right", range=[0, 100], showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        font=dict(family="Inter", size=11, color="#718096"),
    )

    return fig


def create_daily_chart(daily_summaries: list) -> go.Figure:
    """Create a minimalist daily generation summary chart."""
    if not daily_summaries: return go.Figure()
    df = pd.DataFrame(daily_summaries)

    fig = go.Figure(go.Bar(
        x=df["date"],
        y=df["total_generation_kwh"],
        marker=dict(
            color=df["total_generation_kwh"],
            colorscale=[[0, "#FF6B35"], [1, "#F7C948"]],
            line=dict(width=0),
        ),
        hovertemplate="<b>%{x}</b><br>Total: %{y:.1f} kWh<extra></extra>",
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=320,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        showlegend=False,
    )

    return fig


def create_energy_allocation_chart(plan: dict) -> go.Figure:
    """Create a premium donut chart for energy allocation."""
    solar = plan.get("solar_allocation_percent", 65)
    grid = plan.get("grid_import_percent", 15)
    storage = plan.get("storage_usage_percent", 20)

    fig = go.Figure(go.Pie(
        labels=["Solar", "Grid", "Storage"],
        values=[solar, grid, storage],
        hole=0.7,
        marker=dict(
            colors=["#FF6B35", "#3F51B5", "#00C853"],
            line=dict(color="#0B0E14", width=4),
        ),
        textinfo="none",
        hovertemplate="<b>%{label}</b><br>%{value:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=300,
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.1),
        annotations=[dict(text=f"Total<br>100%", x=0.5, y=0.5, font_size=16, showarrow=False, font_family="Outfit")],
    )

    return fig



# ─── Sidebar ───

with st.sidebar:
    st.markdown('<div class="sidebar-logo">SolarAgent.ai</div>', unsafe_allow_html=True)
    
    # API Key
    st.markdown("#### 🔑 API Configuration")
    api_key_input = st.text_input(
        "Google API Key",
        type="password",
        placeholder="Enter your Gemini API key...",
        help="Required for LLM-powered analysis. Get a key at ai.google.dev",
    )

    if api_key_input:
        os.environ["GOOGLE_API_KEY"] = api_key_input

    api_status = get_api_key()
    if api_status:
        st.success("✅ API Key configured")
    else:
        st.warning("⚠️ No API key. System will use statistical fallbacks.")

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # Forecast Parameters
    st.markdown("#### 📊 Forecast Parameters")
    forecast_days = st.slider("Forecast Days", 1, 14, 7, help="Number of days to forecast")

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # Custom Data Upload
    st.markdown("#### 📂 Custom Dataset")
    from models.forecast import save_sample_data
    try:
        sample_csv_path = save_sample_data()
        with open(sample_csv_path, "rb") as f:
            st.download_button(
                "📥 Download Sample CSV",
                f,
                file_name="solar_irradiance_sample.csv",
                mime="text/csv",
                help="Download a sample dataset to see the required column format."
            )
    except Exception as e:
        logger.error(f"Could not generate sample data: {e}")

    uploaded_file = st.file_uploader("Upload Historical Data", type=["csv"], help="Upload CSV containing required columns for forecasting.")

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # Run Agent
    st.markdown("#### 🚀 Execute Pipeline")
    run_clicked = st.button(
        "⚡ Run Grid Optimization Agent",
        type="primary",
        use_container_width=True,
    )

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # System Info
    st.markdown("#### ℹ️ System Info")
    st.caption(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    st.caption("🔧 Agent v1.0.0")
    st.caption("🧠 LangGraph Multi-Node Pipeline")
    st.caption("📚 FAISS RAG (4 knowledge bases)")


# ─── Header ───

# ─── Header ───
st.title("🌞 Solar Grid Optimization Agent")
st.markdown('<p style="color:#718096; font-size:1.1rem; margin-top:-1rem;">Intelligent Renewable Energy Forecasting & Grid Management</p>', unsafe_allow_html=True)
st.markdown('<div style="height: 1px; background: rgba(255,255,255,0.05); margin: 1.5rem 0;"></div>', unsafe_allow_html=True)



# ─── Main Logic ───

if run_clicked:
    custom_df = None
    if uploaded_file is not None:
        try:
            custom_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading uploaded CSV: {e}")
            st.stop()

    # ── Step 1: Generate Forecast ──
    with st.status("⚡ Agent Pipeline Running...", expanded=True) as status:

        st.write("📊 **Step 1/5:** Generating Milestone 1 solar forecast...")
        progress = st.progress(0)
        try:
            forecast_data = generate_forecast(days=forecast_days, custom_df=custom_df)
        except Exception as e:
            status.update(label="❌ Pipeline Error", state="error")
            st.error(f"Forecast Error: {e}")
            st.stop()
        progress.progress(20)
        st.session_state["forecast_data"] = forecast_data
        time.sleep(0.3)

        # ── Step 2: Build FAISS Index ──
        st.write("📚 **Step 2/5:** Building FAISS vector store...")
        try:
            from rag.ingest import build_vectorstore
            build_vectorstore()
        except Exception as e:
            st.warning(f"FAISS index build note: {e}")
        progress.progress(40)
        time.sleep(0.3)

        # ── Step 3: Run Agent Pipeline ──
        st.write("🤖 **Step 3/5:** Running LangGraph agent pipeline...")
        st.write("  → 🔍 Analysis Node → 📚 RAG Retrieval → 🧠 Planning → 📝 Generation")
        progress.progress(50)

        result = run_pipeline(forecast_data)
        progress.progress(80)
        st.session_state["pipeline_result"] = result
        time.sleep(0.3)

        # ── Step 4: Validate Report ──
        st.write("✅ **Step 4/5:** Validating structured report...")
        from utils.error_handling import validate_report_grounding
        report = result.get("final_report", {})
        guidelines = result.get("retrieved_guidelines", [])
        validated_report = validate_report_grounding(report, guidelines)
        st.session_state["validated_report"] = validated_report
        progress.progress(95)
        time.sleep(0.3)

        # ── Step 5: Complete ──
        st.write("🎉 **Step 5/5:** Pipeline complete!")
        progress.progress(100)

        errors = result.get("error_log", [])
        if errors:
            status.update(label="⚡ Pipeline Complete (with warnings)", state="complete")
        else:
            status.update(label="✅ Pipeline Complete — Report Ready", state="complete")


# ─── Display Results ───

if "pipeline_result" in st.session_state:
    result = st.session_state["pipeline_result"]
    forecast_data = st.session_state.get("forecast_data", {})
    report = st.session_state.get("validated_report", result.get("final_report", {}))

    # ── Tabs ──
    tab_dashboard, tab_workflow, tab_report, tab_rag = st.tabs([
        "📊 Dashboard", "🤖 Agent Workflow", "📋 Full Report", "📚 RAG Sources"
    ])

    # ═══════════════════════════════════════════
    # TAB 1: Dashboard
    # ═══════════════════════════════════════════
    with tab_dashboard:
        # KPI Metrics Row
        forecast_summary = report.get("forecast_summary", {})
        risk_analysis = report.get("risk_analysis", {})
        storage_recs = report.get("storage_recommendations", [])
        util_plan = report.get("energy_utilization_plan", {})

        st.markdown('<div style="height: 1rem;"></div>', unsafe_allow_html=True)
        
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            peak = forecast_summary.get("peak_generation_kwh", 0)
            st.markdown(render_kpi(f"{peak:.1f} kW", "Peak Power"), unsafe_allow_html=True)
        with m_col2:
            risk_level = risk_analysis.get("risk_level", result.get("risk_level", "MEDIUM"))
            st.markdown(render_kpi(render_risk_badge(risk_level), "Risk Profile"), unsafe_allow_html=True)
        with m_col3:
            primary_action = storage_recs[0].get("action", "N/A") if storage_recs else "N/A"
            st.markdown(render_kpi(f"🔋 {primary_action}", "Storage Mode"), unsafe_allow_html=True)
        with m_col4:
            savings = util_plan.get("expected_cost_saving_percent", 0)
            st.markdown(render_kpi(f"{savings:.0f}%", "Savings Est."), unsafe_allow_html=True)

        st.markdown('<div style="height: 3rem;"></div>', unsafe_allow_html=True)

        # Main Visualization Grid
        g_col1, g_col2 = st.columns([2, 1])

        with g_col1:
            st.markdown('<p class="section-header">Generation & Variability Forecast</p>', unsafe_allow_html=True)
            with st.container(border=True):
                st.plotly_chart(create_forecast_chart(forecast_data), use_container_width=True)
            
            st.markdown('<div style="height: 2rem;"></div>', unsafe_allow_html=True)
            st.markdown('<p class="section-header">Weekly Cumulative Generation</p>', unsafe_allow_html=True)
            with st.container(border=True):
                st.plotly_chart(create_daily_chart(forecast_data.get("daily_summaries", [])), use_container_width=True)

        with g_col2:
            st.markdown('<p class="section-header">Dynamic Energy Mix</p>', unsafe_allow_html=True)
            with st.container(border=True):
                st.plotly_chart(create_energy_allocation_chart(util_plan if isinstance(util_plan, dict) else {}), use_container_width=True)
            
            st.markdown('<p class="section-header">Model Assurance</p>', unsafe_allow_html=True)
            with st.container(border=True):
                m_data = forecast_data.get("model_metrics", {})
                r2_val = f"{m_data.get('r2_score', 0):.4f}"
                rmse_val = f"{m_data.get('rmse', 0):.4f}"
                conf_val = f"{forecast_summary.get('confidence_level', 92.5):.1f}"
                
                st.markdown(textwrap.dedent(f"""
                <div style="padding: 1rem;">
                    <div style="display:flex; justify-content:space-between; margin-bottom: 0.5rem;">
                        <span style="color:#94A3B8;">Accuracy (R-Squared)</span>
                        <span style="color:#FFFFFF; font-weight:700;">{r2_val}</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; margin-bottom: 0.5rem;">
                        <span style="color:#94A3B8;">RMSE Error</span>
                        <span style="color:#FFFFFF; font-weight:700;">{rmse_val}</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; margin-bottom: 0.5rem;">
                        <span style="color:#94A3B8;">Agent Confidence</span>
                        <span style="color:#FF6B35; font-weight:700;">{conf_val}%</span>
                    </div>
                </div>
                """), unsafe_allow_html=True)


    # ═══════════════════════════════════════════
    # TAB 2: Agent Workflow
    # ═══════════════════════════════════════════
    with tab_workflow:
        st.markdown('<p class="section-header">Agent Node Execution Monitor</p>', unsafe_allow_html=True)

        # High-fidelity status row
        nodes = [
            ("Analysis", "complete"),
            ("Retrieval", "complete"),
            ("Planning", "complete"),
            ("Generation", "complete")
        ]
        
        status_cols = st.columns(4)
        for i, (name, status) in enumerate(nodes):
            with status_cols[i]:
                st.markdown(render_node_status(name, status), unsafe_allow_html=True)

        st.markdown('<div style="height: 2rem;"></div>', unsafe_allow_html=True)

        # Node-by-node outputs with terminal look
        st.markdown('<p style="color:#A0AEC0; font-size: 0.9rem; font-weight:600; text-transform:uppercase; letter-spacing:0.1em;">Agent Logical Reasoning & Execution Logs</p>', unsafe_allow_html=True)

        with st.expander("🔍 Analysis Node: Solar Pattern Recognition", expanded=True):
            st.markdown(textwrap.dedent(f"""
            <div class="terminal-container">
                <span style="color:#FF6B35;">[SYSTEM]</span> Analyzing generational variance...<br>
                <span style="color:#FF6B35;">[DATA]</span> CoV: {forecast_summary.get('variability_index', 0):.4f}<br>
                <span style="color:#FF6B35;">[RISK]</span> Risk Level set to {result.get('risk_level', 'N/A')}<br><br>
                {result.get('analysis_result', '')}
            </div>
            """), unsafe_allow_html=True)

        with st.expander("📚 RAG Node: Regulatory Retrieval"):
            guidelines = result.get("retrieved_guidelines", [])
            st.markdown(f'<div class="terminal-container">', unsafe_allow_html=True)
            st.markdown(f'<span style="color:#00C853;">[QUERY]</span> Searching vector store for grid balancing protocols...<br>', unsafe_allow_html=True)
            st.markdown(f'<span style="color:#00C853;">[FOUND]</span> {len(guidelines)} relevant knowledge chunks identified.<br><br>', unsafe_allow_html=True)
            for g in guidelines[:3]:
                st.markdown(f'<span style="color:#CBD5E0;">» Source: {g.get("source", "Unknown")}</span><br>', unsafe_allow_html=True)
                st.markdown(f'<span style="color:#718096;">{g.get("content", "")[:200]}...</span><br><br>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("🧠 Planning Node: Strategy Formulation"):
            p_data = result.get("energy_plan", {})
            st.markdown(textwrap.dedent(f"""
            <div class="terminal-container">
                <span style="color:#F7C948;">[PLANNER]</span> Formulating energy utilization strategy...<br>
                <span style="color:#F7C948;">[PLANNER]</span> Allocation: {p_data.get('solar_allocation_percent', 0)}% Solar | {p_data.get('storage_usage_percent', 0)}% Storage<br>
                <span style="color:#F7C948;">[PLANNER]</span> Actions: {len(p_data.get('grid_balancing_actions', []))} grid actions scheduled.<br>
            </div>
            """), unsafe_allow_html=True)

        with st.expander("📝 Generation Node: Final Synthesis"):
            st.json(report)


    # ═══════════════════════════════════════════
    # TAB 3: Full Report
    # ═══════════════════════════════════════════
    with tab_report:
        st.markdown('<p class="section-header">Grid Optimization Analysis & Strategy</p>', unsafe_allow_html=True)
        
        # ── 1. Forecast Summary Sheet ──
        with st.container(border=True):
            st.markdown("#### 1. Statistical Generation Forecast")
            fs = report.get("forecast_summary", {})
            if isinstance(fs, dict):
                fcol1, fcol2, fcol3 = st.columns(3)
                with fcol1:
                    st.write(f"**Period:** {fs.get('period', 'N/A')}")
                    st.write(f"**Confidence:** {fs.get('confidence_level', 0):.1f}%")
                with fcol2:
                    st.write(f"**Peak Power:** {fs.get('peak_generation_kwh', 0):.2f} kW")
                    st.write(f"**Variability Index:** {fs.get('variability_index', 0):.3f}")
                with fcol3:
                    st.write(f"**Avg Energy:** {fs.get('avg_generation_kwh', 0):.1f} kWh/day")
                    st.write(f"**Trend:** {fs.get('trend', 'STABLE')}")

        st.markdown('<div style="height: 1.5rem;"></div>', unsafe_allow_html=True)

        # ── 2. Risk & Impact ──
        with st.container(border=True):
            st.markdown("#### 2. Risk Assessment & Interaction Matrix")
            ra = report.get("risk_analysis", {})
            if isinstance(ra, dict):
                st.markdown(f"**Risk Level:** {render_risk_badge(ra.get('risk_level', 'MEDIUM'))}", unsafe_allow_html=True)
                
                rcol1, rcol2 = st.columns(2)
                with rcol1:
                    st.markdown("**Core Risk Factors:**")
                    for factor in ra.get("factors", []):
                        st.markdown(f"  - ⚠️ {factor}")
                with rcol2:
                    st.markdown("**Mitigation Directives:**")
                    for strategy in ra.get("mitigation_strategies", []):
                        st.markdown(f"  - ✅ {strategy}")
                
                st.markdown(f"**Impact Assessment:** {ra.get('impact_assessment', 'N/A')}")

        st.markdown('<div style="height: 1.5rem;"></div>', unsafe_allow_html=True)

        # ── 3. Strategic Actions ──
        with st.container(border=True):
            st.markdown("#### 3. Execution Protocols")
            
            # Grid Actions
            st.markdown("**Grid Balancing Protocols**")
            gba = report.get("grid_balancing_actions", [])
            for action in gba:
                if isinstance(action, dict):
                    priority = action.get("priority", "SCHEDULED")
                    icon = "🔴" if priority == "IMMEDIATE" else "🟡" if priority == "SCHEDULED" else "🟢"
                    st.info(f"{icon} **{action.get('action_type', 'N/A')}** | *{action.get('timeframe', 'N/A')}*\n\n{action.get('description', 'N/A')}")

            # Storage Actions
            st.markdown("**Battery Storage Prescriptions**")
            sr = report.get("storage_recommendations", [])
            for rec in sr:
                if isinstance(rec, dict):
                    action = rec.get("action", "HOLD")
                    st.success(f"🔋 **{action}** | Target SoC: **{rec.get('target_soc_percent', 0)}%**\n\n{rec.get('reasoning', 'N/A')}")

        st.markdown('<div style="height: 1.5rem;"></div>', unsafe_allow_html=True)

        # ── 4. Energy Plan ──
        with st.container(border=True):
            st.markdown("#### 4. Energy Allocation Plan")
            eup = report.get("energy_utilization_plan", {})
            if isinstance(eup, dict):
                ec1, ec2, ec3 = st.columns(3)
                ec1.metric("Solar Focus", f"{eup.get('solar_allocation_percent', 0):.0f}%")
                ec2.metric("Grid Offset", f"{eup.get('grid_import_percent', 0):.0f}%")
                ec3.metric("Storage Use", f"{eup.get('storage_usage_percent', 0):.0f}%")
                
                st.markdown(f"**Demand Response Actions:** {', '.join(eup.get('demand_response_actions', ['None']))}")
                st.markdown(f"**Optimization Intelligence:** {eup.get('optimization_notes', 'N/A')}")

        st.markdown('<div style="height: 2rem;"></div>', unsafe_allow_html=True)

        # ── Download Section ──
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            st.download_button(
                "📥 Export Strategy Report (JSON)",
                json.dumps(report, indent=2, default=str),
                file_name=f"Strategy_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
                use_container_width=True,
            )
        with d_col2:
            report_text = f"SOLAR GRID OPTIMIZATION STRATEGY\n{'-'*30}\n"
            report_text += f"Generated: {report.get('generated_at', 'N/A')}\n\n"
            report_text += f"1. FORECAST: {json.dumps(report.get('forecast_summary', {}), indent=2)}\n\n"
            report_text += f"2. RISK: {json.dumps(report.get('risk_analysis', {}), indent=2)}\n"
            st.download_button(
                "📥 Export Strategy Report (TXT)",
                report_text,
                file_name=f"Strategy_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                use_container_width=True,
            )


    # ═══════════════════════════════════════════
    # TAB 4: RAG Sources
    # ═══════════════════════════════════════════
    with tab_rag:
        st.markdown('<p class="section-header">Retrieved Grid Protocol Documents</p>', unsafe_allow_html=True)
        
        guidelines = result.get("retrieved_guidelines", [])
        if guidelines:
            st.markdown(f"**Semantic Search Engine found {len(guidelines)} highly relevant advisory chunks.**")
            for i, g in enumerate(guidelines, 1):
                source = g.get("source", "Unknown")
                score = float(g.get("score", 0))
                
                # High-fidelity badge within expander
                relevance = "🟢 High" if score < 0.8 else "🟡 Medium" if score < 1.2 else "🔴 Low"
                
                with st.expander(f"CHUNK {i} | {source} | {relevance} Relevance"):
                    st.markdown(textwrap.dedent(f"""
                    <div style="background:rgba(255,107,53,0.02); padding:1rem; border-left: 3px solid #FF6B35; border-radius:4px;">
                        {g.get('content', 'N/A')}
                    </div>
                    <div style="margin-top:1rem; font-size: 0.8rem; color:#718096;">
                        Vector L2 Distance: {score:.4f}
                    </div>
                    """), unsafe_allow_html=True)
        else:
            st.warning("Knowledge base not accessed yet.")


else:
    # ── Landing Page (Empty State) ──
    st.markdown('<div style="height: 2rem;"></div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 3rem;">
            <div style="font-size: 5rem; margin-bottom: 2rem;">☀️</div>
            <h1 style="font-size: 3rem; font-weight: 800; color: #FFFFFF; margin-bottom: 1rem;">Solar Intelligence Ready</h1>
            <p style="color: #94A3B8; font-size: 1.2rem; max-width: 650px; margin: 0 auto; line-height: 1.6;">
                Autonomous energy management at your fingertips. Synchronize generation patterns with grid protocols in real-time.
            </p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            with st.container(border=True):
                st.markdown("### 📊 M1 Forecast")
                st.markdown("Precision hourly generation predictions using high-fidelity ML models.")
        with col2:
            with st.container(border=True):
                st.markdown("### 📚 RAG Protocol")
                st.markdown("Grounding all agent decisions in IEEE standards and regulatory docs.")
        with col3:
            with st.container(border=True):
                st.markdown("### 🤖 Agent Nodes")
                st.markdown("Multi-step LangGraph reasoning chain for grid balancing and storage.")

    st.markdown('<div style="height: 5rem;"></div>', unsafe_allow_html=True)
    st.info("← **System Idle**: Use the sidebar to configure parameters and click **'Run Grid Optimization Agent'** to begin analysis.")

