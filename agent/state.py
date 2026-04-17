"""
LangGraph Agent State Schema.
Defines the TypedDict that flows through all nodes in the workflow.
Uses Annotated types with reducers for proper state accumulation.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class AgentState(TypedDict):
    """
    Central state schema for the Solar Grid Optimization Agent.
    Each node reads from and writes to this shared state.

    Fields:
        forecast_data:        Milestone 1 forecast output (dict with predictions, metrics, metadata)
        analysis_result:      Textual analysis from the Analysis node
        risk_factors:         List of identified risk factors
        risk_level:           Overall risk severity (LOW / MEDIUM / HIGH / CRITICAL)
        retrieved_guidelines: RAG-retrieved document chunks with metadata
        energy_plan:          Structured energy plan from the Planning node
        final_report:         Final structured report (StructuredReport schema)
        current_node:         Name of the currently executing node
        error_log:            Accumulated error messages (append-only via reducer)
        iteration_count:      Safety counter to prevent infinite loops
    """

    # ── Input ──
    forecast_data: dict[str, Any]

    # ── Analysis Node Outputs ──
    analysis_result: str
    risk_factors: list[str]
    risk_level: str  # LOW | MEDIUM | HIGH | CRITICAL

    # ── RAG Node Outputs ──
    retrieved_guidelines: list[dict[str, Any]]

    # ── Planning Node Outputs ──
    energy_plan: dict[str, Any]

    # ── Generation Node Outputs ──
    final_report: dict[str, Any]

    # ── Control Flow ──
    current_node: str
    error_log: Annotated[list[str], operator.add]  # append-only accumulator
    iteration_count: int
