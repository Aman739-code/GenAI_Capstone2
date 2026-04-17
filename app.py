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

# ─── Custom CSS ───
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Hero header */
    .hero-header {
        background: linear-gradient(135deg, #FF6B35 0%, #F7C948 50%, #FF8C42 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0;
        line-height: 1.2;
    }

    .hero-subtitle {
        color: #8B95A5;
        font-size: 1.1rem;
        font-weight: 300;
        margin-top: 0;
    }

    /* Card styling */
    .glass-card {
        background: rgba(26, 29, 41, 0.8);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 107, 53, 0.15);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(255, 107, 53, 0.15);
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(26, 29, 41, 0.9) 0%, rgba(30, 35, 50, 0.9) 100%);
        border: 1px solid rgba(255, 107, 53, 0.2);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #FF6B35, #F7C948);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .metric-label {
        color: #8B95A5;
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.25rem;
    }

    /* Risk badges */
    .risk-low { color: #4CAF50; border-color: #4CAF50; }
    .risk-medium { color: #FF9800; border-color: #FF9800; }
    .risk-high { color: #F44336; border-color: #F44336; }
    .risk-critical { color: #D32F2F; border-color: #D32F2F; background: rgba(211,47,47,0.1); }

    .risk-badge {
        display: inline-block;
        padding: 0.35rem 1rem;
        border-radius: 20px;
        border: 2px solid;
        font-weight: 600;
        font-size: 0.9rem;
        letter-spacing: 0.05em;
    }

    /* Pipeline status */
    .node-status {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        margin: 0.25rem;
        font-size: 0.85rem;
        font-weight: 500;
    }

    .node-active {
        background: rgba(255, 107, 53, 0.15);
        border: 1px solid rgba(255, 107, 53, 0.4);
        color: #FF6B35;
    }

    .node-complete {
        background: rgba(76, 175, 80, 0.15);
        border: 1px solid rgba(76, 175, 80, 0.4);
        color: #4CAF50;
    }

    .node-pending {
        background: rgba(139, 149, 165, 0.1);
        border: 1px solid rgba(139, 149, 165, 0.2);
        color: #8B95A5;
    }

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
""", unsafe_allow_html=True)


# ─── Helper Functions ───

def render_risk_badge(level: str) -> str:
    """Render a styled risk level badge."""
    css_class = f"risk-{level.lower()}" if level else "risk-medium"
    return f'<span class="risk-badge {css_class}">{level}</span>'


def render_metric_card(value: str, label: str) -> str:
    """Render a styled KPI metric card."""
    return f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """


def render_node_status(name: str, status: str) -> str:
    """Render a pipeline node status badge."""
    icons = {"complete": "✅", "active": "⚡", "pending": "⏳", "error": "❌"}
    icon = icons.get(status, "⏳")
    return f'<span class="node-status node-{status}">{icon} {name}</span>'


def create_forecast_chart(forecast_data: dict) -> go.Figure:
    """Create an interactive forecast power output chart."""
    predictions = forecast_data.get("hourly_predictions", [])
    features = forecast_data.get("raw_features", {})
    timestamps = features.get("timestamps", list(range(len(predictions))))

    df = pd.DataFrame({
        "Timestamp": pd.to_datetime(timestamps) if timestamps else range(len(predictions)),
        "Power Output (kW)": predictions,
    })

    fig = go.Figure()

    # Power output area chart
    fig.add_trace(go.Scatter(
        x=df["Timestamp"],
        y=df["Power Output (kW)"],
        fill="tozeroy",
        fillcolor="rgba(255, 107, 53, 0.15)",
        line=dict(color="#FF6B35", width=2),
        name="Predicted Power",
        hovertemplate="<b>%{x}</b><br>Power: %{y:.2f} kW<extra></extra>",
    ))

    # Add cloud cover as secondary axis if available
    if "cloud_cover" in features:
        cloud = features["cloud_cover"]
        fig.add_trace(go.Scatter(
            x=df["Timestamp"],
            y=[c * 100 for c in cloud],
            line=dict(color="#8B95A5", width=1, dash="dot"),
            name="Cloud Cover (%)",
            yaxis="y2",
            hovertemplate="Cloud: %{y:.0f}%<extra></extra>",
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=400,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.05)",
            title="",
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.05)",
            title="Power Output (kW)",
            title_font=dict(color="#FF6B35"),
        ),
        yaxis2=dict(
            title="Cloud Cover (%)",
            title_font=dict(color="#8B95A5"),
            overlaying="y",
            side="right",
            showgrid=False,
            range=[0, 100],
        ),
        hovermode="x unified",
    )

    return fig


def create_daily_chart(daily_summaries: list) -> go.Figure:
    """Create a daily generation summary bar chart."""
    if not daily_summaries:
        return go.Figure()

    df = pd.DataFrame(daily_summaries)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df["date"],
        y=df["total_generation_kwh"],
        marker=dict(
            color=df["total_generation_kwh"],
            colorscale=[[0, "#F44336"], [0.5, "#FF9800"], [1, "#4CAF50"]],
            line=dict(width=0),
        ),
        name="Daily Generation",
        hovertemplate="<b>%{x}</b><br>Generation: %{y:.1f} kWh<extra></extra>",
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=300,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(showgrid=False, title=""),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.05)",
            title="Generation (kWh)",
        ),
        showlegend=False,
    )

    return fig


def create_energy_allocation_chart(plan: dict) -> go.Figure:
    """Create a donut chart for energy allocation."""
    solar = plan.get("solar_allocation_percent", 65)
    grid = plan.get("grid_import_percent", 15)
    storage = plan.get("storage_usage_percent", 20)

    fig = go.Figure(go.Pie(
        labels=["Solar", "Grid Import", "Storage"],
        values=[solar, grid, storage],
        hole=0.55,
        marker=dict(
            colors=["#FF6B35", "#3F51B5", "#4CAF50"],
            line=dict(color="#0E1117", width=3),
        ),
        textinfo="label+percent",
        textfont=dict(size=13, color="white"),
        hovertemplate="<b>%{label}</b><br>%{value:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=300,
        margin=dict(l=20, r=20, t=10, b=10),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.1,
            xanchor="center",
            x=0.5,
        ),
        annotations=[dict(
            text="Energy<br>Mix",
            x=0.5, y=0.5,
            font=dict(size=16, color="#8B95A5"),
            showarrow=False,
        )],
    )

    return fig


# ─── Sidebar ───

with st.sidebar:
    st.markdown('<p class="hero-header" style="font-size:1.5rem;">⚡ Control Panel</p>', unsafe_allow_html=True)
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

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

st.markdown('<p class="hero-header">🌞 Solar Grid Optimization Agent</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">Intelligent Solar Energy Forecasting & Agentic Grid Management</p>',
    unsafe_allow_html=True,
)
st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)


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

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            peak = forecast_summary.get("peak_generation_kwh", 0)
            st.markdown(render_metric_card(f"{peak:.1f} kW", "Peak Generation"), unsafe_allow_html=True)

        with col2:
            risk_level = risk_analysis.get("risk_level", result.get("risk_level", "N/A"))
            st.markdown(render_metric_card(
                render_risk_badge(risk_level), "Risk Level"
            ), unsafe_allow_html=True)

        with col3:
            primary_action = storage_recs[0].get("action", "N/A") if storage_recs else "N/A"
            st.markdown(render_metric_card(f"🔋 {primary_action}", "Storage Action"), unsafe_allow_html=True)

        with col4:
            savings = util_plan.get("expected_cost_saving_percent", 0)
            st.markdown(render_metric_card(f"{savings:.0f}%", "Est. Cost Savings"), unsafe_allow_html=True)

        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

        # Charts
        col_left, col_right = st.columns([2, 1])

        with col_left:
            st.markdown('<p class="section-header">⚡ Power Output Forecast</p>', unsafe_allow_html=True)
            fig = create_forecast_chart(forecast_data)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown('<p class="section-header">📅 Daily Generation Summary</p>', unsafe_allow_html=True)
            daily_fig = create_daily_chart(forecast_data.get("daily_summaries", []))
            st.plotly_chart(daily_fig, use_container_width=True)

        with col_right:
            st.markdown('<p class="section-header">🔄 Energy Allocation</p>', unsafe_allow_html=True)
            alloc_fig = create_energy_allocation_chart(
                util_plan if isinstance(util_plan, dict) else {}
            )
            st.plotly_chart(alloc_fig, use_container_width=True)

            # Model Metrics
            st.markdown('<p class="section-header">🎯 Model Performance</p>', unsafe_allow_html=True)
            metrics = forecast_data.get("model_metrics", {})
            mcol1, mcol2 = st.columns(2)
            with mcol1:
                st.metric("R² Score", f"{metrics.get('r2_score', 0):.4f}")
                st.metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
            with mcol2:
                st.metric("MAE", f"{metrics.get('mae', 0):.4f}")
                confidence = forecast_summary.get("confidence_level", 0)
                st.metric("Confidence", f"{confidence:.1f}%")

    # ═══════════════════════════════════════════
    # TAB 2: Agent Workflow
    # ═══════════════════════════════════════════
    with tab_workflow:
        st.markdown('<p class="section-header">🔄 LangGraph Pipeline Execution</p>', unsafe_allow_html=True)

        # Pipeline visualization
        nodes = [
            ("📊 Analysis", "complete"),
            ("📚 RAG Retrieval", "complete"),
            ("🧠 Planning", "complete"),
            ("📝 Generation", "complete"),
        ]

        errors = result.get("error_log", [])
        if errors:
            nodes.append(("⚠️ Warnings", "error"))

        node_html = " → ".join(render_node_status(name, status) for name, status in nodes)
        st.markdown(f'<div style="text-align:center; padding: 1rem;">{node_html}</div>', unsafe_allow_html=True)

        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

        # Node-by-node outputs
        with st.expander("📊 Analysis Node Output", expanded=True):
            st.markdown(result.get("analysis_result", "No analysis available."))
            st.markdown("**Risk Level:** " + render_risk_badge(result.get("risk_level", "N/A")), unsafe_allow_html=True)
            risk_factors = result.get("risk_factors", [])
            if risk_factors:
                st.markdown("**Risk Factors:**")
                for rf in risk_factors:
                    st.markdown(f"  - {rf}")

        with st.expander("📚 RAG Retrieval Node Output"):
            guidelines = result.get("retrieved_guidelines", [])
            st.markdown(f"**Retrieved {len(guidelines)} guideline chunks**")
            for i, g in enumerate(guidelines[:6], 1):
                st.markdown(f"**[{i}] Source:** `{g.get('source', 'Unknown')}` — Score: `{g.get('score', 'N/A')}`")
                st.caption(g.get("content", "")[:300] + "...")

        with st.expander("🧠 Planning Node Output"):
            plan = result.get("energy_plan", {})
            if plan:
                # Grid Balancing Actions
                actions = plan.get("grid_balancing_actions", [])
                if actions:
                    st.markdown("**Grid Balancing Actions:**")
                    for a in actions:
                        if isinstance(a, dict):
                            priority = a.get("priority", "SCHEDULED")
                            st.markdown(f"- **[{priority}]** {a.get('action_type', 'N/A')}: {a.get('description', '')}")

                # Storage Schedule
                storage = plan.get("storage_schedule", [])
                if storage:
                    st.markdown("**Storage Schedule:**")
                    for s in storage:
                        if isinstance(s, dict):
                            action = s.get("action", "HOLD")
                            st.markdown(f"- 🔋 **{action}** → SoC {s.get('target_soc_percent', 'N/A')}% | {s.get('schedule', 'N/A')}")

        with st.expander("📝 Generation Node Output"):
            st.json(report)

        if errors:
            with st.expander("⚠️ Error Log", expanded=True):
                for err in errors:
                    st.error(err)

    # ═══════════════════════════════════════════
    # TAB 3: Full Report
    # ═══════════════════════════════════════════
    with tab_report:
        st.markdown('<p class="section-header">📋 Grid Optimization Report</p>', unsafe_allow_html=True)

        # ── 1. Forecast Summary ──
        st.markdown("### 1. Forecast Summary")
        fs = report.get("forecast_summary", {})
        if isinstance(fs, dict):
            fcol1, fcol2, fcol3 = st.columns(3)
            with fcol1:
                st.metric("Period", fs.get("period", "N/A"))
                st.metric("Avg Generation", f"{fs.get('avg_generation_kwh', 0):.1f} kWh/day")
            with fcol2:
                st.metric("Peak Generation", f"{fs.get('peak_generation_kwh', 0):.2f} kW")
                st.metric("Min Generation", f"{fs.get('min_generation_kwh', 0):.2f} kW")
            with fcol3:
                st.metric("Variability Index", f"{fs.get('variability_index', 0):.4f}")
                st.metric("Trend", fs.get("trend", "N/A"))

        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

        # ── 2. Risk Analysis ──
        st.markdown("### 2. Risk Analysis")
        ra = report.get("risk_analysis", {})
        if isinstance(ra, dict):
            st.markdown(f"**Risk Level:** {render_risk_badge(ra.get('risk_level', 'N/A'))}", unsafe_allow_html=True)

            st.markdown("**Factors:**")
            for f in ra.get("factors", []):
                st.markdown(f"- ⚠️ {f}")

            st.markdown("**Mitigation Strategies:**")
            for m in ra.get("mitigation_strategies", []):
                st.markdown(f"- ✅ {m}")

            impact = ra.get("impact_assessment", "")
            if impact:
                st.info(f"**Impact Assessment:** {impact}")

        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

        # ── 3. Grid Balancing Actions ──
        st.markdown("### 3. Grid Balancing Actions")
        gba = report.get("grid_balancing_actions", [])
        if gba:
            for action in gba:
                if isinstance(action, dict):
                    priority = action.get("priority", "SCHEDULED")
                    priority_colors = {
                        "IMMEDIATE": "🔴", "SCHEDULED": "🟡", "ADVISORY": "🟢"
                    }
                    icon = priority_colors.get(priority, "🟡")
                    st.markdown(f"""
                    {icon} **{action.get('action_type', 'N/A')}** `[{priority}]`
                    - {action.get('description', 'N/A')}
                    - **Impact:** {action.get('expected_impact', 'N/A')}
                    - **Timeframe:** {action.get('timeframe', 'N/A')}
                    """)

        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

        # ── 4. Storage Recommendations ──
        st.markdown("### 4. Storage Recommendations (Charging/Discharging)")
        sr = report.get("storage_recommendations", [])
        if sr:
            for rec in sr:
                if isinstance(rec, dict):
                    action = rec.get("action", "HOLD")
                    action_icons = {"CHARGE": "🔋⬆️", "DISCHARGE": "🔋⬇️", "HOLD": "🔋⏸️"}
                    icon = action_icons.get(action, "🔋")

                    st.markdown(f"""
                    {icon} **{action}** — Target SoC: **{rec.get('target_soc_percent', 'N/A')}%** | Priority: **{rec.get('priority', 'N/A')}**
                    - **Schedule:** {rec.get('schedule', 'N/A')}
                    - **Reasoning:** {rec.get('reasoning', 'N/A')}
                    """)

        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

        # ── 5. Energy Utilization Plan ──
        st.markdown("### 5. Energy Utilization Plan")
        eup = report.get("energy_utilization_plan", {})
        if isinstance(eup, dict):
            ecol1, ecol2, ecol3 = st.columns(3)
            with ecol1:
                st.metric("☀️ Solar", f"{eup.get('solar_allocation_percent', 0):.1f}%")
            with ecol2:
                st.metric("🔌 Grid Import", f"{eup.get('grid_import_percent', 0):.1f}%")
            with ecol3:
                st.metric("🔋 Storage", f"{eup.get('storage_usage_percent', 0):.1f}%")

            dr_actions = eup.get("demand_response_actions", [])
            if dr_actions:
                st.markdown("**Demand Response Actions:**")
                for action in dr_actions:
                    st.markdown(f"- {action}")

            notes = eup.get("optimization_notes", "")
            if notes:
                st.success(f"💡 **Optimization Notes:** {notes}")

        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

        # ── 6. References ──
        st.markdown("### 6. References")
        refs = report.get("references", [])
        if refs:
            for i, ref in enumerate(refs, 1):
                st.markdown(f"[{i}] 📄 `{ref}`")
        else:
            st.caption("No references cited.")

        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

        # ── Download Buttons ──
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            st.download_button(
                "📥 Download Report (JSON)",
                json.dumps(report, indent=2, default=str),
                file_name=f"solar_grid_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
                use_container_width=True,
            )
        with dcol2:
            # Generate a text version
            report_text = f"""SOLAR GRID OPTIMIZATION REPORT
Generated: {report.get('generated_at', 'N/A')}
Agent Version: {report.get('agent_version', '1.0.0')}

{'='*60}

1. FORECAST SUMMARY
{json.dumps(report.get('forecast_summary', {}), indent=2, default=str)}

2. RISK ANALYSIS
{json.dumps(report.get('risk_analysis', {}), indent=2, default=str)}

3. GRID BALANCING ACTIONS
{json.dumps(report.get('grid_balancing_actions', []), indent=2, default=str)}

4. STORAGE RECOMMENDATIONS
{json.dumps(report.get('storage_recommendations', []), indent=2, default=str)}

5. ENERGY UTILIZATION PLAN
{json.dumps(report.get('energy_utilization_plan', {}), indent=2, default=str)}

6. REFERENCES
{chr(10).join(report.get('references', []))}
"""
            st.download_button(
                "📥 Download Report (TXT)",
                report_text,
                file_name=f"solar_grid_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                use_container_width=True,
            )

    # ═══════════════════════════════════════════
    # TAB 4: RAG Sources
    # ═══════════════════════════════════════════
    with tab_rag:
        st.markdown('<p class="section-header">📚 RAG-Retrieved Grid Guidelines</p>', unsafe_allow_html=True)

        guidelines = result.get("retrieved_guidelines", [])

        if guidelines:
            st.markdown(f"**Total chunks retrieved:** {len(guidelines)}")
            st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

            for i, g in enumerate(guidelines, 1):
                source = g.get("source", "Unknown")
                score = g.get("score", 0)
                content = g.get("content", "N/A")

                # Color based on relevance score (lower = better for FAISS L2)
                if score < 0.8:
                    relevance = "🟢 High Relevance"
                elif score < 1.2:
                    relevance = "🟡 Medium Relevance"
                else:
                    relevance = "🔴 Low Relevance"

                with st.expander(f"[{i}] {source} — {relevance} (Score: {score:.4f})"):
                    st.markdown(content)
        else:
            st.info("No RAG sources retrieved. Run the pipeline to see retrieved guidelines.")

else:
    # ── Empty State ──
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 3rem;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🌞</div>
            <h2 style="color: #FF6B35;">Ready to Optimize Your Grid</h2>
            <p style="color: #8B95A5; font-size: 1.1rem; max-width: 500px; margin: 0 auto;">
                Configure your API key and forecast parameters in the sidebar,
                then click <strong>Run Grid Optimization Agent</strong> to start
                the intelligent analysis pipeline.
            </p>
            <br>
            <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
                <div style="text-align: center;">
                    <div style="font-size: 2rem;">📊</div>
                    <div style="color: #8B95A5; font-size: 0.85rem;">Solar Forecast<br>Analysis</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 2rem;">📚</div>
                    <div style="color: #8B95A5; font-size: 0.85rem;">RAG-Grounded<br>Guidelines</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 2rem;">🧠</div>
                    <div style="color: #8B95A5; font-size: 0.85rem;">Agentic<br>Planning</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 2rem;">📋</div>
                    <div style="color: #8B95A5; font-size: 0.85rem;">Structured<br>Reports</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
