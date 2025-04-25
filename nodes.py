import json
from typing import TypedDict, Annotated, Sequence, Optional
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from models import Intent, PartialClaim  # Import Pydantic models
from agents import intent_agent, extraction_agent  # Import agents


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    intent_analysis: Optional[Intent] = None
    claim_extraction: Optional[PartialClaim] = None
    final_response: Optional[str] = None


async def analyze_message_node(state: AgentState):
    """Analyzes the latest user message for intent and potentially extracts claim details."""
    print("--- Analyzing Message Node ---")
    last_message = state['messages'][-1]
    if not isinstance(last_message, HumanMessage):
        # Should not happen in this simple flow, but good practice
        return {"final_response": "Internal error: Expected user message."}

    user_query = last_message.content

    # 1. Detect Intent
    print(f"Detecting intent for: {user_query}")
    try:
        # Get the result object
        intent_run_result = await intent_agent.run(user_query)
        intent_analysis = intent_run_result.output  # <-- Access the .output attribute
        # Now printing the Intent model
        print(f"Intent detected: {intent_analysis}")
    except Exception as e:
        print(f"Error during intent detection: {e}")
        intent_analysis = None  # Handle potential errors during the agent run

    claim_extraction_result = None
    # 2. Extract Claim Details (if intent is 'create')
    # Make sure intent_result is not None before accessing .action
    if intent_analysis and intent_analysis.action == "create":
        print(f"Extracting claim details for: {user_query}")
        try:
            # <--- Use await
            extraction_run_result = await extraction_agent.run(user_query)
            claim_extraction_result = extraction_run_result.output  # <-- Access the .output attribute
            print(f"Extraction result: {claim_extraction_result}")  # Now printing the PartialClaim model
        except Exception as e:
            print(f"Error during claim extraction: {e}")
            # Keep claim_extraction_result as None
            # generate_response_node will handle this case
            claim_extraction_result = None

    # Handle potential None for intent_result if run fails, though less likely with await
    elif not intent_analysis:
        print("Intent detection failed or returned None.")
        # Set intent to unknown so generate_response handles it gracefully
        intent_analysis = Intent(action="unknown")

    return {
        "intent_analysis": intent_analysis,
        "claim_extraction": claim_extraction_result,
    }


def generate_response_node(state: AgentState):
    """Generates the final response string based on analysis."""
    print("--- Generating Response Node ---")
    intent_analysis = state.get("intent_analysis")
    claim_extraction = state.get("claim_extraction")
    response_str = "Sorry, I couldn't understand that. How can I help with your auto insurance claim?"

    if intent_analysis:
        if intent_analysis.action == "create":
            if claim_extraction:
                # Format the extracted data as pretty JSON markdown
                extracted_data_dict = claim_extraction.model_dump(
                    exclude_none=True)
                # Use default=str for datetime
                pretty_json = json.dumps(
                    extracted_data_dict, indent=2, default=str)
                response_str = f"Okay, I've started a new claim draft with the following details:\n```json\n{pretty_json}\n```\nIs there anything else I should add?"
            else:
                response_str = "Okay, I understand you want to create a claim, but I couldn't extract any details. Could you please provide more information about the incident?"
        elif intent_analysis.action == "retrieve":
            response_str = "Claim retrieval is currently under development. Please check back later."
        # 'unknown' intent will use the default response_str

    print(f"Final response generated: {response_str}")
    return {"final_response": response_str}
