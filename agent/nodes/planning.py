"""
Planning Node — Energy Strategy Formulation.

Uses the analysis results and RAG-retrieved guidelines to formulate
concrete energy management strategies: grid balancing, storage scheduling,
demand response, and utilization optimization.
"""

from __future__ import annotations

import json
import logging

from langchain_google_genai import ChatGoogleGenerativeAI

from agent.state import AgentState
from config.settings import LLM_MODEL, LLM_TEMPERATURE, get_api_key

logger = logging.getLogger(__name__)


def _build_planning_prompt(state: AgentState) -> str:
    """Construct the planning prompt with all available context."""
    forecast_data = state.get("forecast_data", {})
    analysis = state.get("analysis_result", "No analysis available.")
    risk_factors = state.get("risk_factors", [])
    risk_level = state.get("risk_level", "MEDIUM")
    guidelines = state.get("retrieved_guidelines", [])

    # Format guidelines with source citations
    guidelines_text = ""
    for i, g in enumerate(guidelines[:8], 1):
        source = g.get("source", "Unknown")
        content = g.get("content", "")[:600]
        guidelines_text += f"\n### Source [{i}]: {source}\n{content}\n"

    metadata = forecast_data.get("metadata", {})
    daily_summaries = forecast_data.get("daily_summaries", [])

    daily_summary_text = ""
    for d in daily_summaries[:7]:
        daily_summary_text += (
            f"  - {d['date']}: {d['total_generation_kwh']} kWh, "
            f"peak {d['peak_generation_kw']} kW, cloud {d['avg_cloud_cover']:.2f}\n"
        )

    return f"""You are an expert grid operations planner for a solar energy facility. Based on the analysis and retrieved grid guidelines below, create a detailed energy management plan.

## System Information
- Location: {metadata.get('location', 'N/A')}
- System Capacity: {metadata.get('system_capacity_kw', 'N/A')} kW
- Forecast Period: {metadata.get('forecast_start', 'N/A')} to {metadata.get('forecast_end', 'N/A')}

## Daily Generation Forecast
{daily_summary_text}

## Analysis Summary
{analysis[:1500]}

## Risk Level: {risk_level}
## Risk Factors
{chr(10).join(f'- {r}' for r in risk_factors)}

## Retrieved Grid Guidelines
{guidelines_text}

## TASK
Create a comprehensive energy management plan in the following JSON format. All recommendations MUST be grounded in the retrieved guidelines above. Cite the source number [1], [2], etc. in your reasoning.

Return ONLY valid JSON in this exact structure:
{{
    "grid_balancing_actions": [
        {{
            "action_type": "string (e.g., Load Shifting, Frequency Regulation, Curtailment, Demand Response)",
            "priority": "IMMEDIATE or SCHEDULED or ADVISORY",
            "description": "detailed description grounded in guidelines",
            "expected_impact": "expected impact on grid stability",
            "timeframe": "when to execute"
        }}
    ],
    "storage_schedule": [
        {{
            "action": "CHARGE or DISCHARGE or HOLD",
            "target_soc_percent": 0.0,
            "reasoning": "rationale citing guidelines",
            "schedule": "time schedule",
            "priority": "HIGH or MEDIUM or LOW"
        }}
    ],
    "demand_response_actions": [
        "action description 1",
        "action description 2"
    ],
    "optimization_notes": "overall optimization strategy",
    "solar_allocation_percent": 0.0,
    "grid_import_percent": 0.0,
    "storage_usage_percent": 0.0,
    "expected_cost_saving_percent": 0.0
}}"""


def _parse_plan_response(response_text: str) -> dict:
    """Parse the LLM response into a structured plan dictionary."""
    # Try to extract JSON from the response
    text = response_text.strip()

    # Remove markdown code fences if present
    if "```json" in text:
        text = text.split("```json", 1)[1]
        text = text.split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1]
        text = text.split("```", 1)[0]

    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass

    # Fallback plan
    logger.warning("Failed to parse LLM planning response. Using fallback plan.")
    return _get_fallback_plan()


def _get_fallback_plan(risk_level: str = "MEDIUM") -> dict:
    """Return a sensible fallback plan when LLM fails."""
    return {
        "grid_balancing_actions": [
            {
                "action_type": "Frequency Monitoring",
                "priority": "SCHEDULED",
                "description": "Maintain continuous grid frequency monitoring and activate reserves if deviation exceeds ±0.3 Hz.",
                "expected_impact": "Ensures grid stability within regulatory limits",
                "timeframe": "Continuous during forecast period",
            },
            {
                "action_type": "Ramp Rate Control",
                "priority": "SCHEDULED",
                "description": "Limit solar output ramp rate to 10% of capacity per minute per grid balancing guidelines.",
                "expected_impact": "Reduces frequency deviations from solar variability",
                "timeframe": "During daylight generation hours",
            },
        ],
        "storage_schedule": [
            {
                "action": "CHARGE",
                "target_soc_percent": 80.0,
                "reasoning": "Pre-charge during peak solar hours to capture excess generation per storage protocols.",
                "schedule": "10:00 - 14:00 daily",
                "priority": "HIGH",
            },
            {
                "action": "DISCHARGE",
                "target_soc_percent": 30.0,
                "reasoning": "Discharge during evening peak to supplement declining solar per demand response guidelines.",
                "schedule": "17:00 - 21:00 daily",
                "priority": "HIGH",
            },
        ],
        "demand_response_actions": [
            "Activate Category B demand response (commercial HVAC) during anticipated solar shortfalls",
            "Pre-cool buildings during peak solar hours to reduce evening AC demand",
        ],
        "optimization_notes": "Standard solar-storage optimization schedule applied. Monitor real-time conditions for adjustments.",
        "solar_allocation_percent": 65.0,
        "grid_import_percent": 15.0,
        "storage_usage_percent": 20.0,
        "expected_cost_saving_percent": 12.0,
    }


def planning_node(state: AgentState) -> dict:
    """
    LangGraph Planning Node.
    Formulates energy management strategies grounded in RAG-retrieved guidelines.

    Reads:  forecast_data, analysis_result, risk_factors, risk_level, retrieved_guidelines
    Writes: energy_plan, current_node
    """
    logger.info("=== PLANNING NODE: Formulating energy strategies ===")

    try:
        api_key = get_api_key()

        if api_key:
            try:
                llm = ChatGoogleGenerativeAI(
                    model=LLM_MODEL,
                    temperature=LLM_TEMPERATURE,
                    google_api_key=api_key,
                )

                prompt = _build_planning_prompt(state)
                response = llm.invoke(prompt)
                plan = _parse_plan_response(response.content)

            except Exception as e:
                logger.warning(f"LLM planning failed: {e}. Using fallback plan.")
                plan = _get_fallback_plan(state.get("risk_level", "MEDIUM"))
        else:
            logger.info("No API key available. Using fallback plan.")
            plan = _get_fallback_plan(state.get("risk_level", "MEDIUM"))

        logger.info(
            f"Planning complete: {len(plan.get('grid_balancing_actions', []))} balancing actions, "
            f"{len(plan.get('storage_schedule', []))} storage actions"
        )

        return {
            "energy_plan": plan,
            "current_node": "planning",
        }

    except Exception as e:
        logger.error(f"Planning node failed: {e}")
        return {
            "energy_plan": _get_fallback_plan(),
            "current_node": "planning",
            "error_log": [f"Planning node error: {str(e)}"],
        }
