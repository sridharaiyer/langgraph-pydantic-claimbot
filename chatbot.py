# chatbot.py
import asyncio
import json
import os
import uuid  # Import uuid for unique thread IDs

import streamlit as st
from langchain_core.messages import HumanMessage

# Import the LangGraph workflow builder
from langgraph_workflow import build_graph


# --- Streamlit UI ---
st.set_page_config(page_title="ClaimPilot Assistant", page_icon="🚗")
st.title("🚗 ClaimPilot Assistant")
st.caption("Your AI assistant for auto insurance claims.")

# Check for OpenAI API key
if "OPENAI_API_KEY" not in os.environ:
    st.error(
        "OPENAI_API_KEY environment variable not set. Please set it to run the app.")
    st.stop()

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = [
        # Add empty steps list
        {"role": "assistant", "content": "Hi! How can I help you with your auto insurance claim today?", "steps": []}]
# Initialize a unique thread ID if it doesn't exist for the session
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    print(f"New session thread ID: {st.session_state.thread_id}")


# --- Function to format and display JSON within markdown ---
def display_message_content(content):
    """Handles displaying markdown and embedded JSON blocks."""
    if isinstance(content, str) and "```json" in content:
        # Find the start and end of the JSON block
        json_start = content.find("```json") + len("```json\n")
        json_end = content.rfind("```")
        prefix = content[:json_start - len("```json\n")]
        json_content = content[json_start:json_end]
        suffix = content[json_end + len("```"):]

        st.write(prefix.strip())  # Display text before JSON
        try:
            # Parse and display the JSON block using st.json
            parsed_json = json.loads(json_content)
            st.json(parsed_json)
        except json.JSONDecodeError:
            # Fallback if JSON is malformed
            st.code(json_content, language="json")
        st.write(suffix.strip())  # Display text after JSON
    else:
        st.markdown(content)


# --- Display chat messages ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Display intermediate steps if they exist (for assistant messages)
        if message["role"] == "assistant" and message.get("steps"):
            with st.expander("Show Processing Steps", expanded=False):
                for step in message["steps"]:
                    st.markdown(f"- {step}")  # Display steps as a list

        # Display the main content
        display_message_content(message["content"])


# --- Handle user input and graph execution ---
prompt = st.chat_input(
    "What would you like to do? (e.g., 'Start a new claim for my accident')")

if prompt:
    # Add user message to state and display it immediately
    # User messages don't have 'steps'
    st.session_state.messages.append(
        {"role": "user", "content": prompt, "steps": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare graph input
    graph_input = {"messages": [HumanMessage(content=prompt)]}

    # Get the cached graph instance
    graph = build_graph()

    # Define the config using the session's thread_id
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    # Use st.status for the *current* processing, but capture logs separately
    assistant_response = None
    intermediate_steps_log = []  # <-- List to capture steps for this turn
    with st.status("Processing your request...", expanded=True) as status:  # Keep expanded
        try:
            # Define an async function to stream and update UI
            async def stream_and_capture():
                final_state_response = None
                async for chunk in graph.astream(graph_input, config=config, stream_mode="updates"):
                    # chunk is a dictionary where keys are node names
                    for node_name, output_data in chunk.items():
                        log_entry = ""  # Prepare log entry string
                        if node_name == "analyze_message":
                            log_entry = "Analyzing your message for intent..."
                            status.write(log_entry)
                            intermediate_steps_log.append(
                                log_entry)  # Capture log

                            intent = output_data.get("intent_analysis")
                            extraction = output_data.get("claim_extraction")
                            if intent:
                                log_entry = f"Intent Detected: `{intent.action}`"
                                status.write(f"- {log_entry}")
                                intermediate_steps_log.append(
                                    log_entry)  # Capture log
                            if extraction:
                                log_entry = "Extracting claim details..."
                                status.write(f"- {log_entry}")
                                intermediate_steps_log.append(
                                    log_entry)  # Capture log
                                extracted_dict = extraction.model_dump(
                                    exclude_none=True)
                                if extracted_dict:
                                    log_entry = f"Details Found: `{', '.join(extracted_dict.keys())}`"
                                    status.write(f"  - {log_entry}")
                                    intermediate_steps_log.append(
                                        log_entry)  # Capture log
                                else:
                                    log_entry = "No specific details extracted yet."
                                    status.write(f"  - {log_entry}")
                                    intermediate_steps_log.append(
                                        log_entry)  # Capture log

                        elif node_name == "generate_response":
                            log_entry = "Generating final response..."
                            status.write(log_entry)
                            intermediate_steps_log.append(
                                log_entry)  # Capture log
                            final_state_response = output_data.get(
                                "final_response")
                        # Add more elif blocks here for future nodes

                return final_state_response  # Return the final response string

            # Run the async stream function
            assistant_response = asyncio.run(stream_and_capture())

            if assistant_response:
                status.update(label="Processing complete!",
                              state="complete", expanded=False)
            else:
                assistant_response = "Sorry, I couldn't generate a final response."
                status.update(label="Processing ended unexpectedly.",
                              state="error", expanded=True)

        except Exception as e:
            status.update(label="Error processing request",
                          state="error", expanded=True)
            st.error(f"An error occurred: {e}")
            print(f"Error invoking graph: {e}")  # Log error to console
            assistant_response = "I encountered an error. Please try again."
            intermediate_steps_log.append(f"Error: {e}")  # Log error in steps

    # Add assistant response AND captured steps to state
    if assistant_response:  # Only add if we got a valid response
        st.session_state.messages.append(
            # <-- Store steps
            {"role": "assistant", "content": assistant_response, "steps": intermediate_steps_log})

        # Display the final assistant's response in the main chat
        with st.chat_message("assistant"):
            # Display the captured steps in an expander first
            if intermediate_steps_log:
                with st.expander("Show Processing Steps", expanded=False):
                    for step in intermediate_steps_log:
                        st.markdown(f"- {step}")  # Display steps as a list

            # Display the main content (handles JSON etc.)
            display_message_content(assistant_response)
