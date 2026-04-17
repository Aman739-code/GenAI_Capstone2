"""
Analysis Node — Solar Variability & Forecast Pattern Analysis.

This node examines Milestone 1 forecast data to identify:
- Solar generation variability (coefficient of variation)
- Ramp rate patterns
- Risk factors for grid stability
- Anomalous generation patterns
"""

from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI

from agent.state import AgentState
from config.settings import LLM_MODEL, LLM_TEMPERATURE, get_api_key

logger = logging.getLogger(__name__)


def _compute_statistics(forecast_data: dict) -> dict:
    """Compute statistical indicators from forecast data."""
    predictions = np.array(forecast_data.get("hourly_predictions", []))
    daily_summaries = forecast_data.get("daily_summaries", [])

    # Filter to daylight hours (non-zero predictions)
    daylight_preds = predictions[predictions > 0.05]

    stats = {
        "total_hours": len(predictions),
        "generation_hours": len(daylight_preds),
        "mean_output_kw": round(float(np.mean(daylight_preds)), 3) if len(daylight_preds) > 0 else 0,
        "max_output_kw": round(float(np.max(predictions)), 3),
        "min_daylight_output_kw": round(float(np.min(daylight_preds)), 3) if len(daylight_preds) > 0 else 0,
        "std_output_kw": round(float(np.std(daylight_preds)), 3) if len(daylight_preds) > 0 else 0,
    }

    # Coefficient of variation (variability index)
    if stats["mean_output_kw"] > 0:
        stats["cov"] = round(stats["std_output_kw"] / stats["mean_output_kw"], 4)
    else:
        stats["cov"] = 0.0

    # Ramp rates (hour-to-hour changes)
    if len(predictions) > 1:
        ramp_rates = np.diff(predictions)
        stats["max_ramp_up_kw"] = round(float(np.max(ramp_rates)), 3)
        stats["max_ramp_down_kw"] = round(float(np.min(ramp_rates)), 3)
        stats["avg_abs_ramp_kw"] = round(float(np.mean(np.abs(ramp_rates))), 3)
    else:
        stats["max_ramp_up_kw"] = 0
        stats["max_ramp_down_kw"] = 0
        stats["avg_abs_ramp_kw"] = 0

    # Daily variability
    if daily_summaries:
        daily_totals = [d["total_generation_kwh"] for d in daily_summaries]
        stats["daily_mean_kwh"] = round(float(np.mean(daily_totals)), 2)
        stats["daily_std_kwh"] = round(float(np.std(daily_totals)), 2)
        stats["daily_cov"] = round(float(np.std(daily_totals) / max(np.mean(daily_totals), 0.01)), 4)
        stats["best_day_kwh"] = round(float(np.max(daily_totals)), 2)
        stats["worst_day_kwh"] = round(float(np.min(daily_totals)), 2)

    # Cloud cover analysis
    raw_features = forecast_data.get("raw_features", {})
    if "cloud_cover" in raw_features:
        cloud = np.array(raw_features["cloud_cover"])
        stats["avg_cloud_cover"] = round(float(np.mean(cloud)), 4)
        stats["max_cloud_cover"] = round(float(np.max(cloud)), 4)
        stats["high_cloud_hours"] = int(np.sum(cloud > 0.7))

    return stats


def _determine_risk_level(stats: dict) -> tuple[str, list[str]]:
    """Determine risk level and factors based on statistical analysis."""
    risk_factors = []
    score = 0

    # Variability risk
    cov = stats.get("cov", 0)
    if cov > 0.8:
        risk_factors.append(f"Very high output variability (CoV={cov:.2f}, threshold: 0.8)")
        score += 3
    elif cov > 0.5:
        risk_factors.append(f"Elevated output variability (CoV={cov:.2f}, threshold: 0.5)")
        score += 2
    elif cov > 0.3:
        risk_factors.append(f"Moderate output variability (CoV={cov:.2f})")
        score += 1

    # Ramp rate risk
    max_ramp = abs(stats.get("max_ramp_down_kw", 0))
    capacity = stats.get("max_output_kw", 1)
    ramp_pct = (max_ramp / max(capacity, 0.01)) * 100
    if ramp_pct > 50:
        risk_factors.append(f"Severe ramp events detected ({ramp_pct:.0f}% drop in single hour)")
        score += 3
    elif ramp_pct > 30:
        risk_factors.append(f"Significant ramp events ({ramp_pct:.0f}% drop in single hour)")
        score += 2

    # Cloud cover risk
    high_cloud = stats.get("high_cloud_hours", 0)
    total_hours = stats.get("generation_hours", 1)
    cloud_pct = (high_cloud / max(total_hours, 1)) * 100
    if cloud_pct > 40:
        risk_factors.append(f"High cloud cover prevalence ({cloud_pct:.0f}% of daylight hours)")
        score += 2
    elif cloud_pct > 20:
        risk_factors.append(f"Notable cloud cover periods ({cloud_pct:.0f}% of daylight hours)")
        score += 1

    # Daily consistency risk
    daily_cov = stats.get("daily_cov", 0)
    if daily_cov > 0.3:
        risk_factors.append(f"Inconsistent daily generation (day-to-day CoV={daily_cov:.2f})")
        score += 2

    # Low generation risk
    mean_output = stats.get("mean_output_kw", 0)
    if mean_output < capacity * 0.3:
        risk_factors.append(f"Below-expected average generation ({mean_output:.1f} kW vs {capacity:.1f} kW capacity)")
        score += 1

    if not risk_factors:
        risk_factors.append("No significant risk factors identified. Stable generation expected.")

    # Map score to level
    if score >= 7:
        level = "CRITICAL"
    elif score >= 4:
        level = "HIGH"
    elif score >= 2:
        level = "MEDIUM"
    else:
        level = "LOW"

    return level, risk_factors


def analysis_node(state: AgentState) -> dict:
    """
    LangGraph Analysis Node.
    Analyzes solar forecast data to identify variability and risk factors.

    Reads:  forecast_data
    Writes: analysis_result, risk_factors, risk_level, current_node
    """
    logger.info("=== ANALYSIS NODE: Starting solar variability analysis ===")

    try:
        forecast_data = state.get("forecast_data", {})
        if not forecast_data:
            return {
                "analysis_result": "ERROR: No forecast data provided.",
                "risk_factors": ["Missing forecast data"],
                "risk_level": "HIGH",
                "current_node": "analysis",
                "error_log": ["Analysis node: No forecast data in state"],
            }

        # Compute statistics
        stats = _compute_statistics(forecast_data)
        risk_level, risk_factors = _determine_risk_level(stats)

        # Use LLM for natural language analysis
        api_key = get_api_key()
        if api_key:
            try:
                llm = ChatGoogleGenerativeAI(
                    model=LLM_MODEL,
                    temperature=LLM_TEMPERATURE,
                    google_api_key=api_key,
                )

                prompt = f"""You are a solar energy analyst. Analyze the following solar forecast statistics and provide a concise technical analysis.

## Forecast Statistics
- Forecast Period: {forecast_data.get('metadata', {}).get('forecast_start', 'N/A')} to {forecast_data.get('metadata', {}).get('forecast_end', 'N/A')}
- Location: {forecast_data.get('metadata', {}).get('location', 'N/A')}
- System Capacity: {forecast_data.get('metadata', {}).get('system_capacity_kw', 'N/A')} kW
- Mean Output: {stats['mean_output_kw']} kW
- Peak Output: {stats['max_output_kw']} kW
- Variability Index (CoV): {stats['cov']}
- Max Ramp Down: {stats.get('max_ramp_down_kw', 'N/A')} kW/hr
- Daily Mean Generation: {stats.get('daily_mean_kwh', 'N/A')} kWh
- Average Cloud Cover: {stats.get('avg_cloud_cover', 'N/A')}
- High Cloud Hours: {stats.get('high_cloud_hours', 'N/A')} / {stats.get('generation_hours', 'N/A')} daylight hours
- Model R² Score: {forecast_data.get('model_metrics', {}).get('r2_score', 'N/A')}

## Risk Level: {risk_level}
## Risk Factors: {'; '.join(risk_factors)}

Provide a 3-4 paragraph technical analysis covering:
1. Overall generation outlook and capacity utilization
2. Variability patterns and their causes
3. Key risks to grid stability
4. Preliminary recommendations for the planning stage"""

                response = llm.invoke(prompt)
                analysis_text = response.content

            except Exception as e:
                logger.warning(f"LLM analysis failed, using statistical summary: {e}")
                analysis_text = _build_statistical_summary(stats, risk_level, risk_factors, forecast_data)
        else:
            logger.info("No API key available, using statistical summary")
            analysis_text = _build_statistical_summary(stats, risk_level, risk_factors, forecast_data)

        logger.info(f"Analysis complete. Risk level: {risk_level}, Factors: {len(risk_factors)}")

        return {
            "analysis_result": analysis_text,
            "risk_factors": risk_factors,
            "risk_level": risk_level,
            "current_node": "analysis",
        }

    except Exception as e:
        logger.error(f"Analysis node failed: {e}")
        return {
            "analysis_result": f"Analysis failed due to error: {str(e)}",
            "risk_factors": ["Analysis node encountered an error"],
            "risk_level": "MEDIUM",
            "current_node": "analysis",
            "error_log": [f"Analysis node error: {str(e)}"],
        }


def _build_statistical_summary(stats: dict, risk_level: str, risk_factors: list, forecast_data: dict) -> str:
    """Build a statistical summary when LLM is unavailable."""
    metadata = forecast_data.get("metadata", {})
    return f"""## Solar Forecast Analysis — Statistical Summary

**Location:** {metadata.get('location', 'N/A')}
**Period:** {metadata.get('forecast_start', 'N/A')} to {metadata.get('forecast_end', 'N/A')}
**System Capacity:** {metadata.get('system_capacity_kw', 'N/A')} kW

### Generation Overview
The forecast projects a mean daylight output of {stats['mean_output_kw']} kW with peak generation reaching {stats['max_output_kw']} kW. Daily generation averages {stats.get('daily_mean_kwh', 'N/A')} kWh, with the best day producing {stats.get('best_day_kwh', 'N/A')} kWh and the worst day at {stats.get('worst_day_kwh', 'N/A')} kWh.

### Variability Analysis
The coefficient of variation stands at {stats['cov']}, indicating {'high' if stats['cov'] > 0.5 else 'moderate' if stats['cov'] > 0.3 else 'low'} output variability. Maximum ramp-down rate of {stats.get('max_ramp_down_kw', 0)} kW/hr suggests {'significant' if abs(stats.get('max_ramp_down_kw', 0)) > stats['max_output_kw'] * 0.3 else 'manageable'} cloud transient effects.

### Risk Assessment: {risk_level}
{chr(10).join(f'- {rf}' for rf in risk_factors)}

### Model Confidence
The underlying forecast model achieved an R² of {forecast_data.get('model_metrics', {}).get('r2_score', 'N/A')}, RMSE of {forecast_data.get('model_metrics', {}).get('rmse', 'N/A')}, and MAE of {forecast_data.get('model_metrics', {}).get('mae', 'N/A')}.
"""
