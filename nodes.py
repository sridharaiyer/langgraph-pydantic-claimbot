# nodes.py
import json
import httpx  # For making API calls
from typing import TypedDict, Annotated, Sequence, Optional, List, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
# Import Pydantic models
from models import Intent, PartialClaim, SQLQuery, InvalidSQLRequest, SQLResponse, ClaimCreate
from agents import intent_agent, extraction_agent, sql_agent  # Import agents
from synthesizer import synthesize_claim  # Import synthesizer function
from db_utils import execute_sql  # Import database execution function

# --- Configuration ---
API_BASE_URL = "http://127.0.0.1:8000"  # Your running API URL


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # Intent and Extraction
    intent_analysis: Optional[Intent] = None
    claim_extraction: Optional[PartialClaim] = None
    # SQL Retrieval Path
    # To store SQLQuery or InvalidSQLRequest
    sql_response: Optional[SQLResponse] = None
    sql_results: Optional[List[Dict[str, Any]]] = None
    sql_error: Optional[str] = None
    # Claim Creation Path
    synthesized_claim: Optional[ClaimCreate] = None
    post_result: Optional[Dict[str, Any]] = None  # To store API response
    post_error: Optional[str] = None
    # Final Output
    final_response: Optional[str] = None

# --- Nodes ---


async def analyze_message_node(state: AgentState):
    """Analyzes the latest user message for intent and potentially extracts claim details if intent is 'create'."""
    print("--- Analyzing Message Node ---")
    last_message = state['messages'][-1]
    if not isinstance(last_message, HumanMessage):
        # Early exit if not Human
        return {"final_response": "Internal error: Expected user message."}

    user_query = last_message.content
    intent_analysis = None
    claim_extraction_result = None

    # 1. Detect Intent
    print(f"Detecting intent for: {user_query}")
    try:
        intent_run_result = await intent_agent.run(user_query)
        intent_analysis = intent_run_result.output
        print(f"Intent detected: {intent_analysis}")
    except Exception as e:
        print(f"Error during intent detection: {e}")
        # Default to unknown on error
        intent_analysis = Intent(action="unknown")

    # 2. Extract Claim Details (only if intent is 'create')
    if intent_analysis and intent_analysis.action == "create":
        print(f"Extracting claim details for: {user_query}")
        try:
            extraction_run_result = await extraction_agent.run(user_query)
            claim_extraction_result = extraction_run_result.output
            print(f"Extraction result: {claim_extraction_result}")
        except Exception as e:
            print(f"Error during claim extraction: {e}")
            claim_extraction_result = None  # Continue, generate_response will handle

    print("--- Analysis Complete ---")
    return {
        "intent_analysis": intent_analysis,
        "claim_extraction": claim_extraction_result,
    }


async def generate_sql_node(state: AgentState):
    """Generates SQL query if intent is 'retrieve' and query details exist."""
    print("--- Generating SQL Node ---")
    intent_analysis = state.get("intent_analysis")
    sql_response = None

    if intent_analysis and intent_analysis.action == "retrieve" and intent_analysis.query_details:
        query_details = intent_analysis.query_details
        print(f"Generating SQL for details: {query_details}")
        try:
            sql_run_result = await sql_agent.run(query_details)
            # This will be SQLQuery or InvalidSQLRequest
            sql_response = sql_run_result.output
            print(f"SQL Agent Response: {sql_response}")
        except Exception as e:
            print(f"Error during SQL generation: {e}")
            sql_response = InvalidSQLRequest(
                error_message=f"Failed to generate SQL: {e}")
    elif intent_analysis and intent_analysis.action == "retrieve":
        # Handle retrieve intent with no specific details
        sql_response = InvalidSQLRequest(
            error_message="Please provide more specific details for the claim you want to retrieve (e.g., claim ID, policy number, status).")
        print("No specific details for retrieval provided.")
    else:
        # This case should ideally not be reached if routing is correct
        print("SQL generation skipped (intent not 'retrieve' or no details).")

    return {"sql_response": sql_response}


async def execute_sql_node(state: AgentState):
    """Executes the generated SQL query."""
    print("--- Executing SQL Node ---")
    sql_response = state.get("sql_response")
    results = None
    error = None

    if isinstance(sql_response, SQLQuery):
        query = sql_response.sql
        print(f"Executing SQL: {query}")
        try:
            # Assuming execute_sql handles the database interaction
            results, error = execute_sql(query)
            if error:
                print(f"SQL Execution Error: {error}")
            else:
                print(f"SQL Results Count: {len(results)}")
        except Exception as e:
            print(f"Unexpected error executing SQL: {e}")
            error = f"Unexpected error executing query: {e}"
    else:
        # If it's not a SQLQuery object (could be InvalidSQLRequest or None), do nothing
        print("No valid SQL query to execute.")
        # Error message might already be in sql_response if InvalidSQLRequest

    return {"sql_results": results, "sql_error": error}


async def synthesize_claim_node(state: AgentState):
    """Synthesizes missing claim data."""
    print("--- Synthesizing Claim Node ---")
    partial_claim = state.get("claim_extraction")
    synthesized = None
    if partial_claim:
        try:
            synthesized = synthesize_claim(partial_claim)
            print(
                f"Synthesized Claim: {synthesized.model_dump_json(indent=2)}")
        except Exception as e:
            print(f"Error during claim synthesis: {e}")
            # Handle error? Maybe set an error state or just proceed
    else:
        print("No partial claim data to synthesize.")

    return {"synthesized_claim": synthesized}


async def post_claim_node(state: AgentState):
    """Posts the synthesized claim to the API."""
    print("--- Posting Claim Node ---")
    synthesized_claim = state.get("synthesized_claim")
    post_result_data = None
    post_error_data = None

    if synthesized_claim:
        claim_data = synthesized_claim.model_dump(
            mode='json')  # Ensure datetime is serialized
        url = f"{API_BASE_URL}/claims/"
        print(
            f"Posting to {url} with data: {json.dumps(claim_data, indent=2)}")
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=claim_data, timeout=10.0)
                response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
                post_result_data = response.json()
                print(f"API Post Successful: {post_result_data}")
        except httpx.RequestError as e:
            post_error_data = f"API Request Error: Could not connect to {e.request.url!r} - {e}"
            print(post_error_data)
        except httpx.HTTPStatusError as e:
            post_error_data = f"API Error: Status {e.response.status_code} for {e.request.url!r}. Response: {e.response.text}"
            print(post_error_data)
        except Exception as e:
            post_error_data = f"Unexpected error posting claim: {e}"
            print(post_error_data)
    else:
        post_error_data = "No synthesized claim data to post."
        print(post_error_data)

    return {"post_result": post_result_data, "post_error": post_error_data}


def generate_response_node(state: AgentState):
    """Generates the final response based on the completed path."""
    print("--- Generating Final Response Node ---")
    intent_analysis = state.get("intent_analysis")
    response_str = "Sorry, I couldn't process your request completely. How else can I help?"  # Default error

    if intent_analysis:
        action = intent_analysis.action
        if action == "create":
            post_result = state.get("post_result")
            post_error = state.get("post_error")
            synthesized = state.get("synthesized_claim")
            if post_error:
                response_str = f"I tried to create the claim, but encountered an error: {post_error}. The synthesized details were:\n```json\n{json.dumps(synthesized.model_dump(exclude_none=True, mode='json'), indent=2) if synthesized else '{}'}\n```"
            elif post_result and 'id' in post_result:
                claim_id = post_result['id']
                response_str = f"Successfully created claim with ID: `{claim_id}`. Here are the full details:\n```json\n{json.dumps(post_result, indent=2, default=str)}\n```"
            elif synthesized:  # POST might have succeeded but returned unexpected data
                response_str = f"Claim was synthesized, but confirmation from the API was unclear. Details:\n```json\n{json.dumps(synthesized.model_dump(exclude_none=True, mode='json'), indent=2)}\n```"
            else:  # Should not happen if routing is correct, but handles case where synthesis failed
                response_str = "I understood you want to create a claim, but couldn't finalize the details or submit it."

        elif action == "retrieve":
            sql_response = state.get("sql_response")
            sql_results = state.get("sql_results")
            sql_error = state.get("sql_error")

            if isinstance(sql_response, InvalidSQLRequest):
                response_str = sql_response.error_message
            elif sql_error:
                response_str = f"I tried to retrieve the claims, but encountered a database error: {sql_error}"
            # Check if list is not None (can be empty)
            elif sql_results is not None:
                if sql_results:
                    # Format results - maybe as a list or simple table in markdown
                    formatted_results = "\n".join(
                        [f"- Claim ID: `{res.get('id', 'N/A')}`, Status: `{res.get('status', 'N/A')}`, Holder: `{res.get('policy_holder_name', 'N/A')}`" for res in sql_results])
                    response_str = f"Found the following claim(s):\n{formatted_results}"
                else:
                    response_str = "I couldn't find any claims matching your criteria."
            else:  # SQL was generated but execution node didn't run or failed unexpectedly
                response_str = "I generated a query for your request, but couldn't retrieve the results."

        elif action == "unknown":
            response_str = "Okay, how can I help you specifically with creating or retrieving an auto insurance claim?"
        # Add other intent actions if needed

    else:  # If intent analysis itself failed
        response_str = "Sorry, I had trouble understanding your initial request."

    print(f"Final response generated: {response_str}")
    return {"final_response": response_str}
