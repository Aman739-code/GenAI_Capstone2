"""
RAG Retrieval Node — Grid Guideline Retrieval via FAISS.

This node constructs targeted queries from the analysis results
and retrieves relevant grid operation guidelines from the FAISS vector store.
"""

from __future__ import annotations

import logging

from agent.state import AgentState
from rag.retriever import retrieve_multi_query

logger = logging.getLogger(__name__)


def _build_retrieval_queries(analysis_result: str, risk_factors: list[str], risk_level: str) -> list[str]:
    """
    Construct targeted retrieval queries based on analysis findings.
    Different risk profiles trigger different query strategies.
    """
    queries = []

    # Base queries always included
    queries.append("solar energy grid balancing strategies and frequency regulation")
    queries.append("battery storage charging and discharging recommendations for solar variability")

    # Risk-level-specific queries
    if risk_level in ("HIGH", "CRITICAL"):
        queries.append("emergency grid stability measures for high solar variability")
        queries.append("curtailment protocols and demand response activation during grid stress")
        queries.append("battery state of charge management during critical grid conditions")
    elif risk_level == "MEDIUM":
        queries.append("moderate solar variability management and ramp rate control")
        queries.append("optimizing battery storage scheduling for variable solar generation")
    else:
        queries.append("standard solar operations and efficiency optimization best practices")

    # Risk-factor-specific queries
    for factor in risk_factors:
        factor_lower = factor.lower()
        if "ramp" in factor_lower:
            queries.append("managing solar ramp events and fast frequency response requirements")
        if "cloud" in factor_lower:
            queries.append("cloud cover impact on solar generation and smoothing strategies")
        if "variability" in factor_lower or "cov" in factor_lower:
            queries.append("coefficient of variation solar output and grid integration standards")
        if "generation" in factor_lower and "below" in factor_lower:
            queries.append("low solar generation periods and backup power strategies")

    # Regulatory compliance query
    queries.append("IEEE 1547 interconnection requirements and FERC Order 2222 compliance")

    # Deduplicate while preserving order
    seen = set()
    unique_queries = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            unique_queries.append(q)

    logger.info(f"Built {len(unique_queries)} retrieval queries for risk level: {risk_level}")
    return unique_queries


def rag_retrieval_node(state: AgentState) -> dict:
    """
    LangGraph RAG Retrieval Node.
    Retrieves relevant grid guidelines from the FAISS vector store.

    Reads:  analysis_result, risk_factors, risk_level
    Writes: retrieved_guidelines, current_node
    """
    logger.info("=== RAG RETRIEVAL NODE: Querying FAISS vector store ===")

    try:
        analysis_result = state.get("analysis_result", "")
        risk_factors = state.get("risk_factors", [])
        risk_level = state.get("risk_level", "MEDIUM")

        # Build targeted queries
        queries = _build_retrieval_queries(analysis_result, risk_factors, risk_level)

        # Determine retrieval depth based on risk level
        k_per_query = 4 if risk_level in ("HIGH", "CRITICAL") else 3

        # Retrieve documents
        guidelines = retrieve_multi_query(queries, k=k_per_query)

        if not guidelines:
            logger.warning("No guidelines retrieved. Using fallback.")
            guidelines = [{
                "content": "Standard grid operation guidelines: Maintain frequency within ±0.5 Hz. "
                           "Keep battery SoC between 20-80%. Follow IEEE 1547 standards for "
                           "interconnection. Activate demand response for generation shortfalls >30%.",
                "source": "fallback_guidelines",
                "score": 0.0,
            }]

        logger.info(f"Retrieved {len(guidelines)} unique guideline chunks")

        return {
            "retrieved_guidelines": guidelines,
            "current_node": "rag_retrieval",
        }

    except Exception as e:
        logger.error(f"RAG retrieval node failed: {e}")
        return {
            "retrieved_guidelines": [{
                "content": "Retrieval error fallback: Apply standard grid management practices. "
                           "Maintain reserves at 15% of peak load. Monitor frequency continuously. "
                           "Deploy storage for variability smoothing.",
                "source": "error_fallback",
                "score": 0.0,
            }],
            "current_node": "rag_retrieval",
            "error_log": [f"RAG retrieval error: {str(e)}"],
        }
