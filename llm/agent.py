from __future__ import annotations
"""
CrimeVision-QA — LangGraph ReAct Agent

Routes a query through: route → retrieve → reason → respond.
For the 'react' strategy, allows up to MAX_ITERATIONS retrieval rounds.
For all other strategies, runs exactly one retrieval round.

Usage:
    import asyncio
    from llm.agent import run_agent

    result = asyncio.run(run_agent("What happened?", "Assault001"))
    print(result["answer"])
"""

import sys
from pathlib import Path
from typing import TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm.config import video_library_col
from llm.query_model.router import RouterOutput, route_query
from llm.query_model.reasoner import reasoner
from llm.retreival_2 import (
    hybrid_search_frames,
    hybrid_search_transcripts,
    time_windowed_search,
)

_MAX_ITERATIONS = 3  # prevents infinite ReAct loops


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    query: str
    video_id: str
    strategy: str
    router_output: RouterOutput | None
    frame_results: list[dict]
    transcript_results: list[dict]
    video_metadata: dict | None
    iteration: int
    final_answer: dict | None


# ---------------------------------------------------------------------------
# Graph nodes (synchronous — LangGraph wraps them)
# ---------------------------------------------------------------------------

def _node_route(state: AgentState) -> AgentState:
    """Classify query intent."""
    router_out = route_query(state["query"], video_id=state["video_id"])
    return {**state, "router_output": router_out}


def _node_retrieve(state: AgentState) -> AgentState:
    """Fetch relevant context based on router intent."""
    r: RouterOutput = state["router_output"]
    video_id = state["video_id"]
    query = r.search_query
    frame_results: list[dict] = []
    transcript_results: list[dict] = []
    video_metadata = None

    if r.intent == "FIND_FRAME":
        frame_results = hybrid_search_frames(query, video_id=video_id, k=6)

    elif r.intent == "FIND_AUDIO":
        transcript_results = hybrid_search_transcripts(query, video_id=video_id, k=6)

    elif r.intent == "SUMMARIZE_WINDOW" and r.time_range:
        window = time_windowed_search(
            query,
            video_id=video_id,
            start_time=r.time_range["start"],
            end_time=r.time_range["end"],
        )
        frame_results = window["frames"]
        transcript_results = window["transcripts"]

    elif r.intent == "COUNT":
        # Fetch more frames for counting
        frame_results = hybrid_search_frames(query, video_id=video_id, k=10)

    elif r.intent == "FIND_VIDEO_META":
        video_metadata = video_library_col.find_one(
            {"video_id": video_id}, {"_id": 0}
        )

    else:
        # Fallback: search both
        frame_results = hybrid_search_frames(query, video_id=video_id, k=5)
        transcript_results = hybrid_search_transcripts(query, video_id=video_id, k=3)

    return {
        **state,
        "frame_results": frame_results,
        "transcript_results": transcript_results,
        "video_metadata": video_metadata,
    }


def _node_reason(state: AgentState) -> AgentState:
    """Synthesise answer from retrieved context."""
    context: list[dict] = state["frame_results"] + state["transcript_results"]

    # Inject video metadata as a pseudo-document if present
    if state["video_metadata"]:
        context.insert(0, {"metadata": state["video_metadata"]})

    result = reasoner.reason(
        query=state["query"],
        context=context,
        strategy=state["strategy"],
        video_id=state["video_id"],
    )
    return {**state, "final_answer": result, "iteration": state["iteration"] + 1}


def _needs_more_info(state: AgentState) -> bool:
    """For ReAct: decide if another retrieval round is needed."""
    if state["strategy"] != "react":
        return False
    if state["iteration"] >= _MAX_ITERATIONS:
        return False
    answer = (state["final_answer"] or {}).get("answer", "")
    # Simple heuristic: if answer says "not enough information", try again
    return "not enough" in answer.lower() or "cannot determine" in answer.lower()


# ---------------------------------------------------------------------------
# Build and run the graph
# ---------------------------------------------------------------------------

def _build_graph():
    """Build the LangGraph StateGraph."""
    from langgraph.graph import StateGraph, END

    graph = StateGraph(AgentState)

    graph.add_node("route", _node_route)
    graph.add_node("retrieve", _node_retrieve)
    graph.add_node("reason", _node_reason)

    graph.set_entry_point("route")
    graph.add_edge("route", "retrieve")
    graph.add_edge("retrieve", "reason")

    # Conditional edge: loop back for ReAct, else end
    def _after_reason(state: AgentState) -> str:
        if _needs_more_info(state):
            return "retrieve"
        return END

    graph.add_conditional_edges("reason", _after_reason, {"retrieve": "retrieve", END: END})

    return graph.compile()


_compiled_graph = None


def _get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = _build_graph()
    return _compiled_graph


async def run_agent(
    query: str,
    video_id: str,
    strategy: str = "zero_shot",
) -> dict:
    """Run the full RAG agent pipeline asynchronously.

    Returns:
        {"answer": str, "timestamps": list[float], "sources": list[dict], "strategy_used": str}
    """
    initial_state: AgentState = {
        "query": query,
        "video_id": video_id,
        "strategy": strategy,
        "router_output": None,
        "frame_results": [],
        "transcript_results": [],
        "video_metadata": None,
        "iteration": 0,
        "final_answer": None,
    }

    graph = _get_graph()
    final_state = await graph.ainvoke(initial_state)
    return final_state["final_answer"] or {
        "answer": "No answer could be generated.",
        "timestamps": [],
        "sources": [],
        "strategy_used": strategy,
    }


def run_agent_sync(query: str, video_id: str, strategy: str = "zero_shot") -> dict:
    """Synchronous wrapper around run_agent for use in CLI/scripts."""
    import asyncio
    return asyncio.run(run_agent(query, video_id, strategy))
