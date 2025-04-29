# langgraph_workflow.py
import streamlit as st
from typing import TypedDict, Annotated, Sequence, Optional, Literal  # Added Literal
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# Import Pydantic models used in state or routing
from models import Intent, SQLQuery, InvalidSQLRequest  # Import SQL models

# Import all node functions and the AgentState
from nodes import (
    AgentState,
    analyze_message_node,
    generate_sql_node,
    execute_sql_node,
    synthesize_claim_node,
    post_claim_node,
    generate_response_node,
)

# --- Routing Functions ---


def route_after_analysis(state: AgentState) -> Literal["generate_sql", "synthesize_claim", "generate_response", "__end__"]:
    """Routes based on the detected intent."""
    print("--- Routing after Analysis ---")
    intent_analysis = state.get("intent_analysis")
    if intent_analysis:
        action = intent_analysis.action
        print(f"Routing based on action: {action}")
        if action == "create":
            # Check if essential info for synthesis is missing (optional refinement)
            # if not state.get("claim_extraction"):
            #     return "generate_response" # Or a new node to ask for more info
            return "synthesize_claim"
        elif action == "retrieve":
            return "generate_sql"
        else:  # unknown
            return "generate_response"
    else:
        print("No intent found, ending.")
        return END  # Or route to a specific error handling node


def route_after_sql_generation(state: AgentState) -> Literal["execute_sql", "generate_response"]:
    """Routes based on whether SQL generation was successful."""
    print("--- Routing after SQL Generation ---")
    sql_response = state.get("sql_response")
    if isinstance(sql_response, SQLQuery):
        print("Routing to execute SQL.")
        return "execute_sql"
    else:  # InvalidSQLRequest or None
        print("Routing to generate response (SQL invalid or not generated).")
        return "generate_response"


# --- LangGraph Workflow ---
@st.cache_resource
def build_graph():
    """
    Builds the LangGraph workflow with conditional routing.
    Cached by Streamlit to avoid rebuilding on every interaction.
    """
    print("--- Building Graph ---")
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("analyze_message", analyze_message_node)
    workflow.add_node("generate_sql", generate_sql_node)
    workflow.add_node("execute_sql", execute_sql_node)
    workflow.add_node("synthesize_claim", synthesize_claim_node)
    workflow.add_node("post_claim", post_claim_node)
    workflow.add_node("generate_response", generate_response_node)

    # Define edges and conditional routing
    workflow.add_edge(START, "analyze_message")

    # Conditional routing after analysis
    workflow.add_conditional_edges(
        "analyze_message",
        route_after_analysis,
        {
            "generate_sql": "generate_sql",
            "synthesize_claim": "synthesize_claim",
            "generate_response": "generate_response",
            END: END  # Route directly to END if no intent found
        }
    )

    # Conditional routing after SQL generation
    workflow.add_conditional_edges(
        "generate_sql",
        route_after_sql_generation,
        {
            "execute_sql": "execute_sql",
            "generate_response": "generate_response"  # Go to response if SQL invalid
        }
    )

    # Edges after SQL execution or claim posting lead to final response
    workflow.add_edge("execute_sql", "generate_response")
    workflow.add_edge("synthesize_claim", "post_claim")
    workflow.add_edge("post_claim", "generate_response")

    # Final response node leads to the end
    workflow.add_edge("generate_response", END)

    # Compile with MemorySaver
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)
    return graph
