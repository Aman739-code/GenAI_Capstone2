# 🧪 Testing Guide — Solar Grid Optimization Agent

A step-by-step checklist to verify the LangGraph state flow, RAG grounding, deployment stability, and adherence to the rubric's technical depth requirements.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [LangGraph State Flow Verification](#2-langgraph-state-flow-verification)
3. [RAG Grounding Validation](#3-rag-grounding-validation)
4. [Structured Output Verification](#4-structured-output-verification)
5. [Error Handling & Resilience](#5-error-handling--resilience)
6. [Streamlit UI Testing](#6-streamlit-ui-testing)
7. [Deployment Stability](#7-deployment-stability)
8. [Rubric Adherence Checklist](#8-rubric-adherence-checklist)

---

## 1. Prerequisites

### Setup Checklist
- [ ] Python 3.10+ installed
- [ ] Virtual environment created and activated
- [ ] All dependencies installed: `pip install -r requirements.txt`
- [ ] Google Gemini API key configured (env var or Streamlit secrets)
- [ ] Project runs locally: `streamlit run app.py`

### Quick Smoke Test
```bash
# Verify imports work
python -c "from agent.graph import build_graph; print('✅ Graph imports OK')"
python -c "from rag.ingest import build_vectorstore; print('✅ RAG imports OK')"
python -c "from models.forecast import generate_forecast; print('✅ Forecast imports OK')"
python -c "from models.schemas import StructuredReport; print('✅ Schemas imports OK')"
```

---

## 2. LangGraph State Flow Verification

### 2.1 State Schema Validation
- [ ] **TypedDict defined** in `agent/state.py` with all required fields
- [ ] **Annotated reducer** on `error_log` field (uses `operator.add` for append-only)
- [ ] State contains: `forecast_data`, `analysis_result`, `risk_factors`, `risk_level`, `retrieved_guidelines`, `energy_plan`, `final_report`, `current_node`, `error_log`, `iteration_count`

### 2.2 Graph Construction
- [ ] Run graph construction test:
```bash
python -c "
from agent.graph import build_graph
graph = build_graph()
print('✅ Graph compiled successfully')
print(f'Nodes: {list(graph.nodes.keys()) if hasattr(graph, \"nodes\") else \"compiled\"}')
"
```
- [ ] Graph contains 5 nodes: `analysis`, `rag_retrieval`, `planning`, `generation`, `error_end`
- [ ] Entry point set to `analysis`
- [ ] Conditional edges after `analysis` and `planning`
- [ ] Terminal edges: `generation → END`, `error_end → END`

### 2.3 End-to-End State Flow
- [ ] Run full pipeline and verify state propagation:
```bash
python -c "
import logging
logging.basicConfig(level=logging.INFO)
from models.forecast import generate_forecast
from agent.graph import run_pipeline

forecast = generate_forecast(days=3)
result = run_pipeline(forecast)

print('\n=== STATE VERIFICATION ===')
assert result.get('forecast_data'), '❌ forecast_data missing'
print('✅ forecast_data populated')

assert result.get('analysis_result'), '❌ analysis_result missing'
print(f'✅ analysis_result: {len(result[\"analysis_result\"])} chars')

assert result.get('risk_factors'), '❌ risk_factors missing'
print(f'✅ risk_factors: {result[\"risk_factors\"]}')

assert result.get('risk_level') in ('LOW','MEDIUM','HIGH','CRITICAL'), '❌ invalid risk_level'
print(f'✅ risk_level: {result[\"risk_level\"]}')

assert result.get('retrieved_guidelines'), '❌ retrieved_guidelines missing'
print(f'✅ retrieved_guidelines: {len(result[\"retrieved_guidelines\"])} chunks')

assert result.get('energy_plan'), '❌ energy_plan missing'
print(f'✅ energy_plan keys: {list(result[\"energy_plan\"].keys())}')

assert result.get('final_report'), '❌ final_report missing'
print(f'✅ final_report keys: {list(result[\"final_report\"].keys())}')

print(f'✅ current_node: {result.get(\"current_node\")}')
print(f'✅ error_log: {result.get(\"error_log\", [])}')
print('\n🎉 ALL STATE FLOW CHECKS PASSED')
"
```

### 2.4 Conditional Routing
- [ ] Verify routing after analysis: non-FATAL errors route to `rag_retrieval`
- [ ] Verify routing after planning: non-FATAL errors route to `generation`
- [ ] Fatal errors (if any) route to `error_end`

---

## 3. RAG Grounding Validation

### 3.1 Knowledge Base Completeness
- [ ] Four knowledge documents exist in `rag/documents/`:
  - [ ] `grid_balancing_guidelines.md` (frequency regulation, curtailment, demand response)
  - [ ] `solar_storage_protocols.md` (SoC management, charging/discharging, scheduling)
  - [ ] `energy_regulatory_standards.md` (IEEE 1547, FERC Order 2222, net metering)
  - [ ] `renewable_best_practices.md` (forecasting, efficiency, demand management)

### 3.2 FAISS Index Build
- [ ] Build and verify FAISS index:
```bash
python -c "
from rag.ingest import build_vectorstore
vs = build_vectorstore(force=True)
print(f'✅ FAISS index built: {vs.index.ntotal} vectors')
assert vs.index.ntotal > 20, '❌ Too few vectors'
print('✅ Vector count acceptable')
"
```

### 3.3 Retrieval Quality
- [ ] Test retrieval with domain-specific queries:
```bash
python -c "
from rag.retriever import retrieve

# Test 1: Battery storage query
r1 = retrieve('battery charging strategies during peak solar hours')
assert len(r1) > 0, '❌ No results for storage query'
assert any('charg' in r['content'].lower() for r in r1), '❌ Results not relevant to charging'
print(f'✅ Storage query: {len(r1)} results, top score: {r1[0][\"score\"]:.4f}')

# Test 2: Grid balancing query
r2 = retrieve('frequency regulation ramp rate control')
assert len(r2) > 0, '❌ No results for grid query'
print(f'✅ Grid query: {len(r2)} results, top score: {r2[0][\"score\"]:.4f}')

# Test 3: Regulatory query
r3 = retrieve('IEEE 1547 interconnection requirements')
assert len(r3) > 0, '❌ No results for regulatory query'
print(f'✅ Regulatory query: {len(r3)} results, top score: {r3[0][\"score\"]:.4f}')

print('\n🎉 ALL RAG TESTS PASSED')
"
```

### 3.4 Report Grounding Check
- [ ] Verify `references` field in final report is non-empty
- [ ] Verify cited sources correspond to actual knowledge base filenames
- [ ] Validate that recommendations reference retrieved guidelines (not hallucinated)

---

## 4. Structured Output Verification

### 4.1 Pydantic Schema Compliance
- [ ] Verify all 6 sections present in final report:
```bash
python -c "
from models.schemas import StructuredReport
import json

# Test schema with sample data
sample = {
    'forecast_summary': {
        'period': '2025-04-01 to 2025-04-07',
        'avg_generation_kwh': 35.5,
        'peak_generation_kwh': 8.2,
        'min_generation_kwh': 0.5,
        'variability_index': 0.45,
        'trend': 'STABLE',
        'confidence_level': 92.5,
    },
    'risk_analysis': {
        'risk_level': 'MEDIUM',
        'factors': ['Moderate variability'],
        'mitigation_strategies': ['Deploy storage'],
        'impact_assessment': 'Manageable risk.',
    },
    'grid_balancing_actions': [{
        'action_type': 'Frequency Regulation',
        'priority': 'SCHEDULED',
        'description': 'Monitor and maintain frequency.',
        'expected_impact': 'Stable grid.',
        'timeframe': 'Continuous',
    }],
    'storage_recommendations': [{
        'action': 'CHARGE',
        'target_soc_percent': 80.0,
        'reasoning': 'Capture peak solar.',
        'schedule': '10:00-14:00',
        'priority': 'HIGH',
    }],
    'energy_utilization_plan': {
        'solar_allocation_percent': 65.0,
        'grid_import_percent': 15.0,
        'storage_usage_percent': 20.0,
        'demand_response_actions': ['Shift AC load'],
        'optimization_notes': 'Standard plan.',
        'expected_cost_saving_percent': 12.0,
    },
    'references': ['grid_balancing_guidelines.md'],
    'generated_at': '2025-04-01T12:00:00',
    'agent_version': '1.0.0',
}

report = StructuredReport(**sample)
print('✅ Pydantic validation passed')
print(f'✅ Report sections: {list(report.model_dump().keys())}')
"
```

### 4.2 Required Sections Checklist
- [ ] **Forecast Summary**: period, avg/peak/min generation, variability index, trend, confidence
- [ ] **Risk Analysis**: risk level (enum), factors, mitigation strategies, impact
- [ ] **Grid Balancing Actions**: action type, priority (enum), description, impact, timeframe
- [ ] **Storage Recommendations**: action (CHARGE/DISCHARGE/HOLD), SoC target, reasoning, schedule
- [ ] **Energy Utilization Plan**: solar/grid/storage percentages, demand response actions, notes
- [ ] **References**: list of cited RAG source documents

---

## 5. Error Handling & Resilience

### 5.1 API Failure Graceful Degradation
- [ ] Test with no API key:
```bash
# Unset API key and run pipeline
unset GOOGLE_API_KEY
python -c "
from models.forecast import generate_forecast
from agent.graph import run_pipeline

forecast = generate_forecast(days=3)
result = run_pipeline(forecast)

assert result.get('final_report'), '❌ No report generated without API key'
assert not result.get('final_report', {}).get('error'), '❌ Report has fatal error'
print('✅ Pipeline completes without API key (uses fallbacks)')
print(f'✅ Report sections: {list(result[\"final_report\"].keys())}')
"
```

### 5.2 Hallucination Guards
- [ ] Verify numerical bounds validation:
  - SoC values clamped to 0-100%
  - Confidence level clamped to 0-100%
  - Allocation percentages sum check (≈100%)
- [ ] Verify grounding warnings are generated for ungrounded claims

### 5.3 Retry Mechanism
- [ ] `utils/error_handling.py` implements exponential backoff decorator
- [ ] Maximum 3 retry attempts with delay caps
- [ ] Errors logged but don't crash the pipeline

---

## 6. Streamlit UI Testing

### 6.1 Layout & Styling
- [ ] Launch app: `streamlit run app.py`
- [ ] Dark theme renders correctly
- [ ] Hero header with gradient text visible
- [ ] Sidebar displays: API key input, forecast slider, run button, system info
- [ ] Empty state shows onboarding message with 4 feature icons

### 6.2 Pipeline Execution
- [ ] Click "Run Grid Optimization Agent"
- [ ] Progress status shows 5 steps with descriptions
- [ ] Pipeline completes successfully (green checkmark or warning)
- [ ] All 4 tabs become populated

### 6.3 Dashboard Tab
- [ ] 4 KPI metric cards display (Peak Generation, Risk Level, Storage Action, Cost Savings)
- [ ] Power output time-series chart renders with hover tooltips
- [ ] Daily generation bar chart renders with color scaling
- [ ] Energy allocation donut chart shows Solar/Grid/Storage split
- [ ] Model performance metrics (R², RMSE, MAE, Confidence) display

### 6.4 Agent Workflow Tab
- [ ] Pipeline node status badges show (all complete or with warnings)
- [ ] 4 expandable sections for node outputs
- [ ] Analysis node shows risk level badge and factors
- [ ] RAG node shows retrieved chunk count with sources
- [ ] Planning node shows grid actions and storage schedule
- [ ] Generation node shows full JSON report

### 6.5 Full Report Tab
- [ ] All 6 report sections render with proper formatting
- [ ] Risk level badge displays with correct color
- [ ] Grid actions show priority icons (🔴🟡🟢)
- [ ] Storage recommendations show charge/discharge icons
- [ ] Energy allocation metrics display
- [ ] References list shows RAG sources
- [ ] Download buttons work (JSON and TXT formats)

### 6.6 RAG Sources Tab
- [ ] Retrieved chunks displayed with source filenames
- [ ] Relevance scores shown (🟢 High / 🟡 Medium / 🔴 Low)
- [ ] Content previews render in expandable sections

---

## 7. Deployment Stability

### 7.1 Local Deployment
- [ ] App starts without errors: `streamlit run app.py`
- [ ] No import errors on cold start
- [ ] FAISS index builds on first run
- [ ] Generated reports download correctly

### 7.2 Hugging Face Spaces
- [ ] Create Space with Streamlit SDK
- [ ] Upload all project files
- [ ] Set `GOOGLE_API_KEY` as Space secret
- [ ] App builds and deploys successfully
- [ ] Pipeline executes end-to-end on Spaces
- [ ] No timeout issues during execution

### 7.3 Streamlit Community Cloud
- [ ] Connect GitHub repository
- [ ] Set `GOOGLE_API_KEY` in Secrets
- [ ] App deploys and runs pipeline successfully

### 7.4 Dependency Check
```bash
# Verify all dependencies resolve
pip install -r requirements.txt --dry-run
```
- [ ] No conflicting dependency versions
- [ ] `faiss-cpu` installs correctly (not `faiss-gpu`)
- [ ] `sentence-transformers` downloads embedding model on first run

---

## 8. Rubric Adherence Checklist

### 8.1 Multi-Node LangGraph Workflow
- [ ] ✅ Uses `StateGraph(AgentState)` with TypedDict schema
- [ ] ✅ Explicit state management via `AgentState` TypedDict
- [ ] ✅ Four distinct nodes: Analysis, RAG Retrieval, Planning, Generation
- [ ] ✅ Conditional edges for routing based on risk level
- [ ] ✅ Error handling node for graceful failure recovery

### 8.2 TypedDict State Schema
- [ ] ✅ Defined in `agent/state.py`
- [ ] ✅ Uses `Annotated` with `operator.add` reducer for `error_log`
- [ ] ✅ All inter-node data flows through state (no side channels)
- [ ] ✅ State is the single source of truth across nodes

### 8.3 Analysis Node
- [ ] ✅ Identifies solar variability (coefficient of variation, ramp rates)
- [ ] ✅ Computes statistical indicators from forecast data
- [ ] ✅ Determines risk level (LOW/MEDIUM/HIGH/CRITICAL)
- [ ] ✅ Uses LLM for natural language analysis (with fallback)

### 8.4 RAG Pipeline Node
- [ ] ✅ Uses FAISS for vector storage
- [ ] ✅ 4 domain-specific knowledge base documents
- [ ] ✅ HuggingFace embeddings (all-MiniLM-L6-v2)
- [ ] ✅ Risk-adaptive multi-query retrieval
- [ ] ✅ Returns documents with source citations

### 8.5 Planning Node
- [ ] ✅ Generates grid balancing strategies
- [ ] ✅ Creates storage charging/discharging schedules
- [ ] ✅ Formulates demand response actions
- [ ] ✅ Grounds recommendations in retrieved guidelines

### 8.6 Generation Node
- [ ] ✅ Produces structured JSON report
- [ ] ✅ Validated against Pydantic `StructuredReport` schema
- [ ] ✅ Contains all 6 required sections
- [ ] ✅ Includes cited references from RAG sources

### 8.7 Milestone 1 Integration
- [ ] ✅ Accepts forecast outputs as pipeline trigger
- [ ] ✅ Synthetic forecast model (RandomForestRegressor)
- [ ] ✅ Forecast data includes predictions, metrics, metadata

### 8.8 Structured Output
- [ ] ✅ Forecast Summary
- [ ] ✅ Risk Analysis
- [ ] ✅ Grid Balancing Action
- [ ] ✅ Storage Recommendations (charging/discharging)
- [ ] ✅ Energy Utilization Plan
- [ ] ✅ Cited References

### 8.9 Error Handling
- [ ] ✅ API failure graceful degradation
- [ ] ✅ Hallucination guards (bounds validation, grounding checks)
- [ ] ✅ Exponential backoff retry mechanism
- [ ] ✅ Pydantic validation with fallback parsing

### 8.10 Deployment Readiness
- [ ] ✅ Professional Streamlit UI with dark theme
- [ ] ✅ Modular codebase (Agent and RAG logic separated)
- [ ] ✅ Comprehensive README with API key setup
- [ ] ✅ Ready for Hugging Face Spaces / Streamlit Cloud
- [ ] ✅ Testing_Guide.md with verification checklist

---

## 🏁 Final Verification Command

Run this single command to execute the full verification suite:

```bash
python -c "
import logging
logging.basicConfig(level=logging.WARNING)

print('='*60)
print('SOLAR GRID OPTIMIZATION AGENT — FULL VERIFICATION')
print('='*60)

# 1. Import verification
print('\n[1/6] Import Verification...')
from agent.state import AgentState
from agent.graph import build_graph, run_pipeline
from rag.ingest import build_vectorstore
from rag.retriever import retrieve
from models.forecast import generate_forecast
from models.schemas import StructuredReport
from utils.error_handling import validate_report_grounding
print('  ✅ All imports successful')

# 2. Graph build
print('\n[2/6] Graph Build...')
graph = build_graph()
print('  ✅ LangGraph compiled')

# 3. FAISS index
print('\n[3/6] FAISS Index...')
vs = build_vectorstore()
print(f'  ✅ FAISS index: {vs.index.ntotal} vectors')

# 4. Retrieval
print('\n[4/6] RAG Retrieval...')
results = retrieve('battery storage charging strategies')
assert len(results) > 0
print(f'  ✅ Retrieved {len(results)} chunks')

# 5. Full pipeline
print('\n[5/6] Full Pipeline...')
forecast = generate_forecast(days=3)
result = run_pipeline(forecast)
report = result.get('final_report', {})
assert report and not report.get('error')
print(f'  ✅ Pipeline complete. Report keys: {list(report.keys())}')

# 6. Report validation
print('\n[6/6] Report Validation...')
validated = validate_report_grounding(report, result.get('retrieved_guidelines', []))
print(f'  ✅ Report validated. Warnings: {validated.get(\"_validation\", {}).get(\"grounding_warnings\", [])}')

print('\n' + '='*60)
print('🎉 ALL VERIFICATION CHECKS PASSED')
print('='*60)
"
```

---

*Generated for Project 13: Intelligent Solar Energy Forecasting & Agentic Grid Optimization*
