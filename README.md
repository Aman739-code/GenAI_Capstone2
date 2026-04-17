# 🌞 Intelligent Solar Energy Forecasting & Agentic Grid Optimization

An autonomous grid management assistant that evolves a Milestone 1 ML forecasting model into a multi-node LangGraph agentic workflow with RAG-grounded decision-making and structured report generation.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_AI-FF6B35)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Store-4285F4)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Streamlit Professional UI                     │
│  ┌──────────┐ ┌──────────────┐ ┌────────────┐ ┌─────────────┐  │
│  │Dashboard │ │Agent Workflow│ │Full Report  │ │ RAG Sources │  │
│  └──────────┘ └──────────────┘ └────────────┘ └─────────────┘  │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                   LangGraph Multi-Node Pipeline                  │
│                                                                  │
│  ┌──────────┐   ┌──────────────┐   ┌─────────┐   ┌──────────┐  │
│  │ Analysis │──▶│RAG Retrieval │──▶│Planning │──▶│Generation│  │
│  │   Node   │   │    Node      │   │  Node   │   │   Node   │  │
│  └──────────┘   └──────┬───────┘   └─────────┘   └──────────┘  │
│                         │                                        │
│              ┌──────────▼──────────┐                             │
│              │  FAISS Vector Store │                             │
│              │  (4 Knowledge Bases)│                             │
│              └─────────────────────┘                             │
└──────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│              Milestone 1 Forecast Model (Trigger)                │
│         RandomForestRegressor on Synthetic Solar Data            │
└──────────────────────────────────────────────────────────────────┘
```

### LangGraph State Flow

```
AgentState (TypedDict)
├── forecast_data          ← Input (Milestone 1 output)
├── analysis_result        ← Analysis Node
├── risk_factors           ← Analysis Node
├── risk_level             ← Analysis Node
├── retrieved_guidelines   ← RAG Retrieval Node
├── energy_plan            ← Planning Node
├── final_report           ← Generation Node
├── current_node           ← Control flow
├── error_log              ← Annotated[list, operator.add] (reducer)
└── iteration_count        ← Safety counter
```

---

## 📁 Project Structure

```
milestone2/
├── app.py                      # Streamlit UI entry point
├── requirements.txt            # Dependencies
├── README.md                   # This file
├── Testing_Guide.md            # Testing & verification checklist
├── .streamlit/config.toml      # Dark theme & server config
├── config/
│   └── settings.py             # API keys, model configs, constants
├── agent/                      # 🤖 Agent Logic (separated)
│   ├── state.py                # TypedDict state schema
│   ├── graph.py                # LangGraph workflow definition
│   └── nodes/
│       ├── analysis.py         # Solar variability analysis
│       ├── rag_retrieval.py    # FAISS retrieval pipeline
│       ├── planning.py         # Energy strategy formulation
│       └── generation.py       # Structured report generation
├── rag/                        # 📚 RAG Logic (separated)
│   ├── documents/              # Knowledge base markdown files
│   │   ├── grid_balancing_guidelines.md
│   │   ├── solar_storage_protocols.md
│   │   ├── energy_regulatory_standards.md
│   │   └── renewable_best_practices.md
│   ├── ingest.py               # Document → FAISS indexing
│   └── retriever.py            # Semantic retrieval interface
├── models/
│   ├── forecast.py             # Milestone 1 synthetic model
│   └── schemas.py              # Pydantic output schemas
├── utils/
│   ├── error_handling.py       # API retry, hallucination guards
│   └── helpers.py              # Utility functions
├── data/                       # Sample data directory
└── vectorstore/                # FAISS index (auto-generated)
```

> **Modularity**: Agent logic (`agent/`) and RAG logic (`rag/`) are fully separated into independent packages with clean interfaces.

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **Google Gemini API Key** (free tier available at [ai.google.dev](https://ai.google.dev))

### 1. Clone & Install

```bash
# Clone the repository
git clone <repo-url>
cd milestone2

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key

**Option A: Environment Variable**
```bash
export GOOGLE_API_KEY="your-gemini-api-key-here"
```

**Option B: Streamlit Secrets** (recommended for deployment)
```bash
mkdir -p .streamlit
echo 'GOOGLE_API_KEY = "your-gemini-api-key-here"' > .streamlit/secrets.toml
```

**Option C: In-App Input**
Enter your API key directly in the sidebar when running the app.

> **Note:** The system works without an API key using statistical fallbacks, but LLM-powered analysis provides significantly richer insights.

### 3. Run Locally

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. Use the sidebar to configure parameters and click **Run Grid Optimization Agent**.

---

## 🔑 API Key Setup

| Provider | Model | Setup | Cost |
|----------|-------|-------|------|
| **Google Gemini** (default) | `gemini-2.0-flash` | [ai.google.dev](https://ai.google.dev) | Free tier (60 RPM) |

### Getting a Google Gemini API Key

1. Go to [Google AI Studio](https://ai.google.dev)
2. Sign in with your Google account
3. Click **"Get API Key"** → **"Create API Key"**
4. Copy the key and set it via one of the methods above

---

## 🌐 Deployment

### Hugging Face Spaces

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
2. Select **Streamlit** as the SDK
3. Upload all project files
4. Add `GOOGLE_API_KEY` as a **Space Secret** (Settings → Secrets)
5. The app will auto-deploy

### Streamlit Community Cloud

1. Push code to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo and set `app.py` as the main file
4. Add `GOOGLE_API_KEY` in **Advanced Settings → Secrets**

---

## 📊 Features

### Multi-Node LangGraph Pipeline
- **Analysis Node**: Statistical computation (CoV, ramp rates) + LLM-powered insights
- **RAG Retrieval Node**: Risk-adaptive multi-query FAISS retrieval from 4 knowledge bases
- **Planning Node**: Context-grounded energy strategy formulation
- **Generation Node**: Pydantic-validated structured report with 6 sections

### Structured Report Output
1. **Forecast Summary** — Generation outlook, variability index, trend
2. **Risk Analysis** — Risk level, factors, mitigation strategies
3. **Grid Balancing Actions** — Prioritized actions with timeframes
4. **Storage Recommendations** — Charge/Discharge/Hold with SoC targets
5. **Energy Utilization Plan** — Solar/Grid/Storage allocation percentages
6. **References** — Cited RAG sources for grounding

### Error Resilience
- Exponential backoff retry on API failures (3 attempts)
- Graceful degradation with statistical fallbacks
- Hallucination guards: numerical bounds validation, grounding checks
- Pydantic schema validation with fallback parsing

### Professional UI
- Dark theme with glassmorphism aesthetic
- Interactive Plotly charts (time-series, daily bars, allocation donut)
- Tabbed layout: Dashboard, Agent Workflow, Full Report, RAG Sources
- Downloadable reports (JSON / TXT)

---

## 🧪 Testing

See [Testing_Guide.md](./Testing_Guide.md) for a comprehensive testing checklist covering:
- LangGraph state flow verification
- RAG grounding validation
- Deployment stability checks
- Rubric adherence verification

---

## 📜 License

This project is developed for educational and internship evaluation purposes.
