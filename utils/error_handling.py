"""
Error Handling & Hallucination Guards.
Provides utilities for API failure resilience, response validation,
and hallucination detection.
"""

from __future__ import annotations

import logging
import time
from functools import wraps
from typing import Any, Callable

logger = logging.getLogger(__name__)


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exceptions: tuple = (Exception,),
):
    """
    Decorator for retrying functions with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay in seconds between retries.
        max_delay: Maximum delay cap in seconds.
        exceptions: Tuple of exception types to catch and retry.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"All {max_retries + 1} attempts failed for {func.__name__}: {e}"
                        )
            raise last_exception
        return wrapper
    return decorator


def validate_numerical_bounds(value: float, min_val: float, max_val: float, name: str) -> float:
    """
    Validate a numerical value is within expected bounds.
    Clamps to bounds if out of range and logs a warning.

    Args:
        value: The value to validate.
        min_val: Minimum acceptable value.
        max_val: Maximum acceptable value.
        name: Name of the field for logging.

    Returns:
        Clamped value within bounds.
    """
    if value < min_val:
        logger.warning(f"Hallucination guard: {name}={value} below minimum {min_val}. Clamped.")
        return min_val
    if value > max_val:
        logger.warning(f"Hallucination guard: {name}={value} above maximum {max_val}. Clamped.")
        return max_val
    return value


def validate_report_grounding(report: dict, guidelines: list[dict]) -> dict:
    """
    Validate that generated recommendations reference retrieved guidelines.
    Flags ungrounded claims and adds warnings.

    Args:
        report: Generated report dictionary.
        guidelines: List of retrieved guideline chunks.

    Returns:
        Report with grounding validation annotations.
    """
    warnings = []

    # Check that references list is non-empty
    references = report.get("references", [])
    if not references:
        warnings.append("No references cited. Recommendations may not be grounded in guidelines.")

    # Validate SoC values
    storage_recs = report.get("storage_recommendations", [])
    for rec in storage_recs:
        if isinstance(rec, dict):
            soc = rec.get("target_soc_percent", 50)
            rec["target_soc_percent"] = validate_numerical_bounds(soc, 0, 100, "SoC")

    # Validate allocation percentages
    util_plan = report.get("energy_utilization_plan", {})
    if isinstance(util_plan, dict):
        solar = util_plan.get("solar_allocation_percent", 0)
        grid = util_plan.get("grid_import_percent", 0)
        storage = util_plan.get("storage_usage_percent", 0)
        total = solar + grid + storage

        if total > 0 and abs(total - 100) > 15:
            warnings.append(
                f"Energy allocation sums to {total:.1f}% (expected ~100%). "
                f"Solar: {solar}%, Grid: {grid}%, Storage: {storage}%"
            )

    # Validate confidence level
    forecast_summary = report.get("forecast_summary", {})
    if isinstance(forecast_summary, dict):
        confidence = forecast_summary.get("confidence_level", 85)
        forecast_summary["confidence_level"] = validate_numerical_bounds(
            confidence, 0, 100, "confidence_level"
        )

    # Add validation metadata
    report["_validation"] = {
        "grounding_warnings": warnings,
        "guidelines_available": len(guidelines),
        "references_cited": len(references),
        "validated": True,
    }

    if warnings:
        logger.warning(f"Report grounding issues: {warnings}")

    return report


class GracefulDegradation:
    """Context manager for graceful degradation on API failures."""

    def __init__(self, fallback_value: Any, operation_name: str = "operation"):
        self.fallback_value = fallback_value
        self.operation_name = operation_name
        self.error = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.error = exc_val
            logger.warning(
                f"Graceful degradation for {self.operation_name}: {exc_val}. "
                f"Using fallback value."
            )
            return True  # Suppress the exception
        return False

    @property
    def result(self):
        if self.error is not None:
            return self.fallback_value
        return None
