# langgraph_workflow.py
import streamlit as st  # <-- Import Streamlit for caching
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from nodes import analyze_message_node, generate_response_node, AgentState


# --- LangGraph Workflow ---
# Add Streamlit caching decorator
@st.cache_resource
def build_graph():
    """
    Builds the LangGraph workflow.
    Cached by Streamlit to avoid rebuilding on every interaction.
    """
    print("--- Building Graph ---")  # Add print statement to observe caching
    workflow = StateGraph(AgentState)

    workflow.add_node("analyze_message", analyze_message_node)
    workflow.add_node("generate_response", generate_response_node)

    workflow.add_edge(START, "analyze_message")
    workflow.add_edge("analyze_message", "generate_response")
    workflow.add_edge("generate_response", END)

    # Using MemorySaver for simplicity, stores state in memory.
    memory = MemorySaver()
    # Note: The checkpointer itself isn't cached in a way that persists
    # its internal memory across Streamlit reruns unless managed carefully
    # outside the cached function or using a persistent backend.
    # For this setup, MemorySaver state will be lost on script restart,
    # but the compiled graph structure is cached.
    graph = workflow.compile(checkpointer=memory)
    return graph

# Removed AgentState definition from here as it's imported from nodes.py
