# chatbot.py
import asyncio
import json
import os
import uuid  # Import uuid for unique thread IDs
import pandas as pd  # Import pandas for displaying SQL results

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
        # Add steps and sql_query
        {"role": "assistant", "content": "Hi! How can I help you with your auto insurance claim today?", "steps": [], "sql_query": None}]
# Initialize a unique thread ID if it doesn't exist for the session
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    print(f"New session thread ID: {st.session_state.thread_id}")


# --- Function to format and display content (handles JSON, DataFrames, and Markdown) ---
def display_message_content(content):
    """Handles displaying markdown, embedded JSON blocks, and DataFrames."""
    if isinstance(content, pd.DataFrame):
        st.dataframe(content)  # Use st.dataframe for pandas DataFrames
    elif isinstance(content, list) and all(isinstance(item, dict) for item in content):
        # Handle list of dicts (potentially from SQL) by converting to DataFrame
        try:
            df = pd.DataFrame(content)
            st.dataframe(df)
        except Exception:  # Fallback if DataFrame creation fails
            st.json(content)  # Display as JSON
    elif isinstance(content, str) and content.strip().startswith("```json"):
        json_content = content.strip()[len("```json"):].rstrip("`")
        try:
            parsed_json = json.loads(json_content)
            st.json(parsed_json)
        except json.JSONDecodeError:
            st.code(json_content, language="json")  # Fallback
    elif isinstance(content, str) and "```json" in content:  # Handle JSON within markdown
        json_start = content.find("```json") + len("```json\n")
        json_end = content.rfind("```")
        if json_start < json_end:
            prefix = content[:json_start - len("```json\n")]
            json_content = content[json_start:json_end]
            suffix = content[json_end + len("```"):]
            st.markdown(prefix.strip())
            try:
                parsed_json = json.loads(json_content)
                st.json(parsed_json)
            except json.JSONDecodeError:
                st.code(json_content, language="json")
            st.markdown(suffix.strip())
        else:
            st.markdown(content)
    elif isinstance(content, str):
        st.markdown(content)
    else:
        st.write(content)  # Default fallback


# --- Display chat messages ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Display intermediate steps if they exist
        if message["role"] == "assistant" and message.get("steps"):
            with st.expander("Show Processing Steps", expanded=False):
                for step in message["steps"]:
                    st.markdown(f"- {step}")

        # Display executed SQL query if it exists
        if message["role"] == "assistant" and message.get("sql_query"):
            with st.expander("Executed SQL Query", expanded=False):
                st.code(message["sql_query"], language="sql")

        # Display the main content using the helper
        display_message_content(message["content"])


# --- Handle user input and graph execution ---
prompt = st.chat_input(
    "What would you like to do? (e.g., 'Start a new claim for my accident', 'Show my approved claims')")

if prompt:
    # Add user message to state and display it immediately
    st.session_state.messages.append(
        {"role": "user", "content": prompt, "steps": [], "sql_query": None})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare graph input
    graph_input = {"messages": [HumanMessage(content=prompt)]}

    # Get the cached graph instance
    graph = build_graph()

    # Define the config using the session's thread_id
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    # Use st.status to show intermediate steps
    assistant_response_content = None
    final_sql_results_list = None  # Store SQL results as list of dicts
    executed_sql = None  # Store the executed SQL query
    intermediate_steps_log = []

    with st.status("Processing your request...", expanded=True) as status:
        try:
            # Define an async function to stream and update UI
            async def stream_and_capture():
                final_response = None
                sql_results = None
                sql_query_executed = None  # Local var for executed query in this run

                async for chunk in graph.astream(graph_input, config=config, stream_mode="updates"):
                    for node_name, output_data in chunk.items():
                        log_entry = f"Executing node: `{node_name}`"
                        status.write(log_entry)
                        intermediate_steps_log.append(log_entry)

                        # --- Capture specific data from nodes ---
                        if node_name == "analyze_message":
                            intent = output_data.get("intent_analysis")
                            # ... (logging for intent/extraction as before) ...
                            if intent:
                                log_entry = f"Intent Detected: `{intent.action}`" + (
                                    f" (Details: `{intent.query_details}`)" if intent.query_details else "")
                                status.write(f"- {log_entry}")
                                intermediate_steps_log.append(log_entry)
                            # ... (logging for extraction)

                        elif node_name == "generate_sql":
                            sql_resp = output_data.get("sql_response")
                            # ... (logging for SQL generation as before) ...
                            if isinstance(sql_resp, dict) and sql_resp.get("sql"):
                                log_entry = f"Generated SQL: \n```sql\n{sql_resp['sql']}\n```"
                                status.write(f"- {log_entry}")
                                intermediate_steps_log.append(log_entry)
                            elif isinstance(sql_resp, dict) and sql_resp.get("error_message"):
                                log_entry = f"SQL Generation Failed: `{sql_resp['error_message']}`"
                                status.write(f"- {log_entry}")
                                intermediate_steps_log.append(log_entry)
                            else:
                                log_entry = "SQL generation step completed."
                                status.write(f"- {log_entry}")
                                intermediate_steps_log.append(log_entry)

                        elif node_name == "execute_sql":
                            results = output_data.get("sql_results")
                            error = output_data.get("sql_error")
                            sql_query_executed = output_data.get(
                                "executed_sql_query")  # <-- Capture executed SQL

                            if sql_query_executed:
                                log_entry = f"Executed SQL: \n```sql\n{sql_query_executed}\n```"
                                status.write(f"- {log_entry}")
                                intermediate_steps_log.append(log_entry)
                            if error:
                                log_entry = f"SQL Execution Error: `{error}`"
                                status.write(f"- {log_entry}")
                                intermediate_steps_log.append(log_entry)
                            elif results is not None:
                                log_entry = f"SQL Execution Successful: Found `{len(results)}` record(s)."
                                status.write(f"- {log_entry}")
                                intermediate_steps_log.append(log_entry)
                                sql_results = results  # Store results list
                            else:
                                log_entry = "SQL execution step completed."
                                status.write(f"- {log_entry}")
                                intermediate_steps_log.append(log_entry)

                        elif node_name == "synthesize_claim":
                            # ... (logging for synthesis as before) ...
                            synth_claim = output_data.get("synthesized_claim")
                            if synth_claim:
                                log_entry = "Synthesized missing claim details."
                                status.write(f"- {log_entry}")
                                intermediate_steps_log.append(log_entry)
                            else:
                                log_entry = "Claim synthesis step completed (no data or failed)."
                                status.write(f"- {log_entry}")
                                intermediate_steps_log.append(log_entry)

                        elif node_name == "post_claim":
                            # ... (logging for posting as before) ...
                            post_res = output_data.get("post_result")
                            post_err = output_data.get("post_error")
                            if post_err:
                                log_entry = f"Claim Posting Error: `{post_err}`"
                                status.write(f"- {log_entry}")
                                intermediate_steps_log.append(log_entry)
                            elif post_res:
                                log_entry = f"Claim Posting Successful (Claim ID: `{post_res.get('id', 'N/A')}`)."
                                status.write(f"- {log_entry}")
                                intermediate_steps_log.append(log_entry)
                            else:
                                log_entry = "Claim posting step completed."
                                status.write(f"- {log_entry}")
                                intermediate_steps_log.append(log_entry)

                        elif node_name == "generate_response":
                            log_entry = "Generating final response..."
                            status.write(f"- {log_entry}")
                            intermediate_steps_log.append(log_entry)
                            final_response = output_data.get("final_response")

                # Return final string response, results list, and executed SQL
                return final_response, sql_results, sql_query_executed

            # Run the async stream function
            assistant_response_content, final_sql_results_list, executed_sql = asyncio.run(
                stream_and_capture())

            if assistant_response_content:
                status.update(label="Processing complete!",
                              state="complete", expanded=False)
            else:
                assistant_response_content = "Sorry, I couldn't generate a final response."
                status.update(label="Processing ended unexpectedly.",
                              state="error", expanded=True)

        except Exception as e:
            status.update(label="Error processing request",
                          state="error", expanded=True)
            st.error(f"An error occurred: {e}")
            print(f"Error invoking graph: {e}")
            assistant_response_content = "I encountered an error. Please try again."
            intermediate_steps_log.append(f"Error: {e}")

    # --- Store and Display Final Response ---
    if assistant_response_content:
        display_content = assistant_response_content
        # If SQL results were generated, convert them to a DataFrame for storage/display
        final_sql_dataframe = None
        if final_sql_results_list is not None:
            try:
                final_sql_dataframe = pd.DataFrame(final_sql_results_list)
                display_content = final_sql_dataframe  # Set DataFrame as main content
                # Optional: Change the string message if results were found
                if not final_sql_dataframe.empty:
                    assistant_response_content = f"Found {len(final_sql_dataframe)} claim(s):"
                else:
                    assistant_response_content = "No claims found matching your criteria."

            except Exception as df_error:
                print(f"Error converting SQL results to DataFrame: {df_error}")
                # Keep results as list/dict in content if DataFrame fails
                display_content = final_sql_results_list
                assistant_response_content += " (Could not display results as table)"

        # Store the message, including the executed SQL and the steps
        st.session_state.messages.append({
            "role": "assistant",
            "content": display_content,  # Store DataFrame or original string
            "steps": intermediate_steps_log,
            "sql_query": executed_sql  # Store the executed SQL query
        })

        # Display the final assistant's response in the main chat
        with st.chat_message("assistant"):
            # Display the captured steps in an expander first
            if intermediate_steps_log:
                with st.expander("Show Processing Steps", expanded=False):
                    for step in intermediate_steps_log:
                        st.markdown(f"- {step}")

            # Display the executed SQL query if it exists
            if executed_sql:
                with st.expander("Executed SQL Query", expanded=False):
                    st.code(executed_sql, language="sql")

            # Display the main content (handles DataFrame, JSON, Markdown)
            # Pass DataFrame or string
            display_message_content(display_content)
