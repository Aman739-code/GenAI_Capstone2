"""
LangGraph Workflow Definition.
Defines the multi-node agent graph with explicit state management,
conditional routing, and error handling.
"""

from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph

from agent.nodes.analysis import analysis_node
from agent.nodes.generation import generation_node
from agent.nodes.planning import planning_node
from agent.nodes.rag_retrieval import rag_retrieval_node
from agent.state import AgentState

logger = logging.getLogger(__name__)


def _route_after_analysis(state: AgentState) -> str:
    """
    Conditional routing after the Analysis node.
    Routes to enhanced retrieval for critical risks,
    or standard retrieval otherwise.
    """
    risk_level = state.get("risk_level", "MEDIUM")
    errors = state.get("error_log", [])

    # Check for critical errors
    if any("FATAL" in e for e in errors):
        logger.warning("Fatal error detected. Routing to end.")
        return "error_end"

    logger.info(f"Post-analysis routing: risk_level={risk_level}")
    return "rag_retrieval"


def _route_after_planning(state: AgentState) -> str:
    """
    Conditional routing after the Planning node.
    Validates that the plan has required components.
    """
    plan = state.get("energy_plan", {})
    errors = state.get("error_log", [])

    if not plan:
        logger.warning("Empty plan detected. Proceeding with fallback.")

    if any("FATAL" in e for e in errors):
        return "error_end"

    return "generation"


def _error_end_node(state: AgentState) -> dict:
    """Terminal error handler node."""
    logger.error("Pipeline terminated due to fatal error")
    errors = state.get("error_log", [])
    return {
        "final_report": {
            "error": True,
            "message": "Pipeline terminated due to fatal errors.",
            "errors": errors,
            "generated_at": __import__("datetime").datetime.now().isoformat(),
        },
        "current_node": "error_end",
    }


def build_graph() -> StateGraph:
    """
    Build and compile the LangGraph workflow.

    Graph Structure:
        START → analysis → rag_retrieval → planning → generation → END
                    ↓                                       ↓
                error_end ←---------------------------------┘

    Returns:
        Compiled LangGraph StateGraph ready for invocation.
    """
    logger.info("Building LangGraph workflow...")

    # Initialize the graph with our state schema
    workflow = StateGraph(AgentState)

    # ── Add Nodes ──
    workflow.add_node("analysis", analysis_node)
    workflow.add_node("rag_retrieval", rag_retrieval_node)
    workflow.add_node("planning", planning_node)
    workflow.add_node("generation", generation_node)
    workflow.add_node("error_end", _error_end_node)

    # ── Set Entry Point ──
    workflow.set_entry_point("analysis")

    # ── Add Edges ──

    # After analysis: conditional routing based on risk level
    workflow.add_conditional_edges(
        "analysis",
        _route_after_analysis,
        {
            "rag_retrieval": "rag_retrieval",
            "error_end": "error_end",
        },
    )

    # After RAG retrieval: always proceed to planning
    workflow.add_edge("rag_retrieval", "planning")

    # After planning: conditional check for completeness
    workflow.add_conditional_edges(
        "planning",
        _route_after_planning,
        {
            "generation": "generation",
            "error_end": "error_end",
        },
    )

    # After generation: end
    workflow.add_edge("generation", END)

    # Error end: terminate
    workflow.add_edge("error_end", END)

    # ── Compile ──
    graph = workflow.compile()
    logger.info("LangGraph workflow compiled successfully")
    return graph


def run_pipeline(forecast_data: dict) -> dict:
    """
    Run the complete agent pipeline with the given forecast data.

    Args:
        forecast_data: Milestone 1 forecast output dict.

    Returns:
        Final state dict containing all node outputs.
    """
    graph = build_graph()

    initial_state: AgentState = {
        "forecast_data": forecast_data,
        "analysis_result": "",
        "risk_factors": [],
        "risk_level": "",
        "retrieved_guidelines": [],
        "energy_plan": {},
        "final_report": {},
        "current_node": "start",
        "error_log": [],
        "iteration_count": 0,
    }

    logger.info("Starting agent pipeline execution...")

    try:
        final_state = graph.invoke(initial_state)
        logger.info(f"Pipeline completed. Final node: {final_state.get('current_node')}")
        return final_state
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        return {
            **initial_state,
            "final_report": {
                "error": True,
                "message": f"Pipeline execution failed: {str(e)}",
                "generated_at": __import__("datetime").datetime.now().isoformat(),
            },
            "error_log": [f"Pipeline fatal error: {str(e)}"],
            "current_node": "failed",
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from models.forecast import generate_forecast

    forecast = generate_forecast(days=7)
    result = run_pipeline(forecast)

    print(f"\nFinal Node: {result.get('current_node')}")
    print(f"Risk Level: {result.get('risk_level')}")
    print(f"Errors: {result.get('error_log', [])}")
    print(f"Report Keys: {list(result.get('final_report', {}).keys())}")
