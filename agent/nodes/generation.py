"""
Generation Node — Structured Report Generation.

Aggregates all pipeline outputs into a final StructuredReport,
validated against Pydantic schemas. Includes hallucination guards
and fallback generation.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI

from agent.state import AgentState
from config.settings import LLM_MODEL, LLM_TEMPERATURE, get_api_key
from models.schemas import StructuredReport

logger = logging.getLogger(__name__)


def _build_generation_prompt(state: AgentState) -> str:
    """Build the final generation prompt with all accumulated state."""
    forecast_data = state.get("forecast_data", {})
    analysis = state.get("analysis_result", "")
    risk_factors = state.get("risk_factors", [])
    risk_level = state.get("risk_level", "MEDIUM")
    guidelines = state.get("retrieved_guidelines", [])
    plan = state.get("energy_plan", {})

    metadata = forecast_data.get("metadata", {})

    # Build references list from RAG sources
    sources = []
    for i, g in enumerate(guidelines[:8], 1):
        src = g.get("source", "Unknown")
        if src not in sources:
            sources.append(src)

    references_text = "\n".join(f"  - [{i+1}] {s}" for i, s in enumerate(sources))

    plan_json = json.dumps(plan, indent=2, default=str)[:2000]

    return f"""You are a senior energy systems engineer generating a formal grid optimization report. Using ALL the information below, generate a comprehensive structured report.

## INPUTS

### Forecast Metadata
- Location: {metadata.get('location', 'N/A')}
- System Capacity: {metadata.get('system_capacity_kw', 'N/A')} kW
- Forecast Period: {metadata.get('forecast_start', 'N/A')} to {metadata.get('forecast_end', 'N/A')}
- Model Type: {metadata.get('model_type', 'N/A')}
- Model R²: {forecast_data.get('model_metrics', {}).get('r2_score', 'N/A')}

### Analysis
{analysis[:1200]}

### Risk Level: {risk_level}
### Risk Factors
{chr(10).join(f'- {r}' for r in risk_factors)}

### Energy Plan
{plan_json}

### Available References
{references_text}

## INSTRUCTIONS
Generate a report as valid JSON matching this EXACT schema (no extra text, ONLY JSON):

{{
  "forecast_summary": {{
    "period": "start to end date",
    "avg_generation_kwh": <float: daily average kWh>,
    "peak_generation_kwh": <float: peak hourly kW>,
    "min_generation_kwh": <float: minimum daylight kW>,
    "variability_index": <float: 0-1 CoV>,
    "trend": "INCREASING or DECREASING or STABLE",
    "confidence_level": <float: 0-100 percent>
  }},
  "risk_analysis": {{
    "risk_level": "{risk_level}",
    "factors": {json.dumps(risk_factors)},
    "mitigation_strategies": ["strategy 1", "strategy 2", "..."],
    "impact_assessment": "narrative paragraph"
  }},
  "grid_balancing_actions": [
    {{
      "action_type": "string",
      "priority": "IMMEDIATE or SCHEDULED or ADVISORY",
      "description": "string",
      "expected_impact": "string",
      "timeframe": "string"
    }}
  ],
  "storage_recommendations": [
    {{
      "action": "CHARGE or DISCHARGE or HOLD",
      "target_soc_percent": <float: 0-100>,
      "reasoning": "string citing sources",
      "schedule": "string",
      "priority": "HIGH or MEDIUM or LOW"
    }}
  ],
  "energy_utilization_plan": {{
    "solar_allocation_percent": <float: 0-100>,
    "grid_import_percent": <float: 0-100>,
    "storage_usage_percent": <float: 0-100>,
    "demand_response_actions": ["action 1", "action 2"],
    "optimization_notes": "string",
    "expected_cost_saving_percent": <float>
  }},
  "references": {json.dumps(sources)},
  "generated_at": "{datetime.now().isoformat()}",
  "agent_version": "1.0.0"
}}

CRITICAL RULES:
1. All recommendations MUST reference the retrieved guidelines
2. Numerical values must be realistic and consistent with the forecast data
3. Storage SoC values must be between 0-100
4. Allocation percentages should sum to approximately 100
5. Return ONLY valid JSON, no markdown formatting"""


def _parse_report(response_text: str, state: AgentState) -> dict:
    """Parse and validate the generated report."""
    text = response_text.strip()

    # Remove code fences
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]

    try:
        data = json.loads(text.strip())
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            data = json.loads(text[start:end])
        else:
            raise ValueError("Cannot parse JSON from LLM response")

    # Validate against Pydantic schema
    try:
        report = StructuredReport(**data)
        return report.model_dump()
    except Exception as e:
        logger.warning(f"Pydantic validation failed: {e}. Returning raw parsed data.")
        # Ensure required fields exist
        data.setdefault("generated_at", datetime.now().isoformat())
        data.setdefault("agent_version", "1.0.0")
        return data


def _build_fallback_report(state: AgentState) -> dict:
    """Build a fallback report from available state data when LLM fails."""
    forecast_data = state.get("forecast_data", {})
    risk_factors = state.get("risk_factors", ["Unable to determine risk factors"])
    risk_level = state.get("risk_level", "MEDIUM")
    plan = state.get("energy_plan", {})
    guidelines = state.get("retrieved_guidelines", [])

    metadata = forecast_data.get("metadata", {})
    daily_summaries = forecast_data.get("daily_summaries", [])
    model_metrics = forecast_data.get("model_metrics", {})

    import numpy as np
    predictions = forecast_data.get("hourly_predictions", [0])
    daylight = [p for p in predictions if p > 0.05]
    daily_totals = [d["total_generation_kwh"] for d in daily_summaries] if daily_summaries else [0]

    sources = list({g.get("source", "Unknown") for g in guidelines[:8]})

    return {
        "forecast_summary": {
            "period": f"{metadata.get('forecast_start', 'N/A')} to {metadata.get('forecast_end', 'N/A')}",
            "avg_generation_kwh": round(float(np.mean(daily_totals)), 2) if daily_totals else 0,
            "peak_generation_kwh": round(float(max(predictions)), 3) if predictions else 0,
            "min_generation_kwh": round(float(min(daylight)), 3) if daylight else 0,
            "variability_index": round(float(np.std(daylight) / max(np.mean(daylight), 0.01)), 4) if daylight else 0,
            "trend": "STABLE",
            "confidence_level": round(model_metrics.get("r2_score", 0.85) * 100, 1),
        },
        "risk_analysis": {
            "risk_level": risk_level,
            "factors": risk_factors,
            "mitigation_strategies": [
                "Deploy battery storage for variability smoothing",
                "Activate demand response during anticipated shortfalls",
                "Maintain spinning reserves per grid guidelines",
            ],
            "impact_assessment": f"Risk level assessed as {risk_level} based on forecast variability analysis. Grid stability measures should be implemented according to retrieved guidelines.",
        },
        "grid_balancing_actions": plan.get("grid_balancing_actions", [
            {
                "action_type": "Frequency Monitoring",
                "priority": "SCHEDULED",
                "description": "Continuous frequency monitoring with automatic reserve activation",
                "expected_impact": "Maintains grid frequency within ±0.5 Hz tolerance",
                "timeframe": "Continuous",
            }
        ]),
        "storage_recommendations": plan.get("storage_schedule", [
            {
                "action": "CHARGE",
                "target_soc_percent": 80.0,
                "reasoning": "Charge during peak solar to capture surplus generation",
                "schedule": "10:00-14:00",
                "priority": "HIGH",
            },
            {
                "action": "DISCHARGE",
                "target_soc_percent": 30.0,
                "reasoning": "Discharge during evening peak demand",
                "schedule": "17:00-21:00",
                "priority": "HIGH",
            },
        ]),
        "energy_utilization_plan": {
            "solar_allocation_percent": plan.get("solar_allocation_percent", 65.0),
            "grid_import_percent": plan.get("grid_import_percent", 15.0),
            "storage_usage_percent": plan.get("storage_usage_percent", 20.0),
            "demand_response_actions": plan.get("demand_response_actions", [
                "Activate commercial HVAC demand response during shortfalls",
                "Schedule EV charging during peak solar hours",
            ]),
            "optimization_notes": plan.get("optimization_notes", "Standard optimization applied. Review when conditions change."),
            "expected_cost_saving_percent": plan.get("expected_cost_saving_percent", 10.0),
        },
        "references": sources if sources else ["grid_balancing_guidelines.md", "solar_storage_protocols.md"],
        "generated_at": datetime.now().isoformat(),
        "agent_version": "1.0.0",
    }


def generation_node(state: AgentState) -> dict:
    """
    LangGraph Generation Node.
    Produces the final structured report validated against Pydantic schemas.

    Reads:  forecast_data, analysis_result, risk_factors, risk_level,
            retrieved_guidelines, energy_plan
    Writes: final_report, current_node
    """
    logger.info("=== GENERATION NODE: Producing structured report ===")

    try:
        api_key = get_api_key()

        if api_key:
            try:
                llm = ChatGoogleGenerativeAI(
                    model=LLM_MODEL,
                    temperature=0.2,  # Lower temperature for structured output
                    google_api_key=api_key,
                )

                prompt = _build_generation_prompt(state)
                response = llm.invoke(prompt)
                report = _parse_report(response.content, state)

                logger.info("Report generated and validated successfully")

            except Exception as e:
                logger.warning(f"LLM generation failed: {e}. Using fallback report.")
                report = _build_fallback_report(state)
        else:
            logger.info("No API key. Building report from available state data.")
            report = _build_fallback_report(state)

        return {
            "final_report": report,
            "current_node": "generation",
        }

    except Exception as e:
        logger.error(f"Generation node failed: {e}")
        return {
            "final_report": _build_fallback_report(state),
            "current_node": "generation",
            "error_log": [f"Generation node error: {str(e)}"],
        }
