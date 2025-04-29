# agents.py
from pydantic_ai import Agent as PydanticAIAgent
from models import Intent, PartialClaim, SQLResponse  # Import Pydantic models
from db_utils import DB_SCHEMA  # Import database schema

LLM_MODEL = "gpt-4.1-nano"

# Agent 1a: Intent Detection
intent_agent = PydanticAIAgent(
    LLM_MODEL,
    output_type=Intent,
    system_prompt="""
    You are an expert at understanding user messages related to auto insurance claims.
    Your task is to determine the user's intent based on their message regarding an auto insurance claim.
    Classify the intent as one of: 'create', 'retrieve', or 'unknown'.

    - 'create': The user wants to report a new incident or start a new claim. 
    Phrases like "I got into an accident", "Someone hit my car", "Need to file a claim".
    - 'retrieve': The user wants to find existing claim information. 
    Phrases like "What's the status of my claim?", "Find claim 123", "Show me claims for John Doe", 
    "List all approved claims".
    - 'unknown': The intent is unclear, conversational, or not related to creating/retrieving claims.

    If the intent is 'retrieve', extract any specific details mentioned for filtering, 
    such as claim ID, policy holder name, status, company, etc., into the 'query_details' field. 
    If no specific details are mentioned for retrieval (e.g., "show me my claims"), 
    'query_details' should be null.
    If the intent is 'create' or 'unknown', 'query_details' must be null.

    Respond ONLY with the JSON object matching the 'Intent' schema.

    Examples:
    User: I hit a deer this morning.
    Output: {"action": "create", "query_details": null}

    User: Can you find the claim for policy number POL-123456?
    Output: {"action": "retrieve", "query_details": "policy number POL-123456"}

    User: What's the status of claim CLM-9876543210?
    Output: {"action": "retrieve", "query_details": "claim ID CLM-9876543210"}

    User: Show me all claims handled by Ryan Cooper.
    Output: {"action": "retrieve", "query_details": "adjuster Ryan Cooper"}

    User: List claims for Beta Insurance that are in progress.
    Output: {"action": "retrieve", "query_details": "Beta Insurance claims with status Repair in Progress"}

    User: Thanks!
    Output: {"action": "unknown", "query_details": null}

    User: Tell me about my options.
    Output: {"action": "unknown", "query_details": null}

    """,
    # instrument=True # Uncomment if using Logfire/LangSmith
)

# Agent 1b: Claim Extraction (only called if intent is 'create')
extraction_agent = PydanticAIAgent(
    LLM_MODEL,
    output_type=PartialClaim,
    system_prompt="""
    You are a helpful assistant trained to extract structured information from user-provided auto accident 
    descriptions.
    Analyze the user's message and extract ONLY the details they explicitly mention regarding the claim.
    Extract details like:
    - policy_holder_name
    - policy_number
    - vehicle_make, vehicle_model, vehicle_year
    - incident_date (infer datetime for relative terms like 'yesterday'/'this morning' 
        within the last 24 hours; assume 2025 for partial dates like 'January 15th')
    - incident_description
    - adjuster_name
    - status
    - company
    - claim_office
    - point_of_impact (e.g., 'front bumper', 'driver side door')

    Use the provided 'PartialClaim' schema for the output.

    Given a natural language description of a car accident, determine the most accurate 
    Point of Impact (POI) on the user's vehicle based on the details provided.
    You must choose exactly one of the following pre-defined POI values:
    - Front  
    - Left Front  
    - Right Front  
    - Left Side  
    - Right Side  
    - Left Rear  
    - Right Rear  
    - Rear  
    - Top  
    - Undercarriage / Bottom  
    - Multiple Points / Multiple Areas  
    - Unknown / Not Specified

    Guidelines:
    Infer direction based on terms like "driver side" (Left in US), "passenger side" (Right), 
    "rear-ended", "hit from the front", "sideswiped", "t-boned", etc.
    If multiple areas are clearly impacted, return Multiple Points / Multiple Areas.
    If the description is too vague or lacks detail, return Unknown / Not Specified.

    Do NOT invent or fill in any details that are not present in the user's text. 
    Output null or omit fields that are not mentioned.

    Example 1:
    User: Hi, my name is John Carter. My car was rear-ended yesterday evening while I was stopped at a 
    red light. It’s a 2020 Honda Accord. My policy number is HN12345678.
    Output: {
    "policy_holder_name": "John Carter",
    "policy_number": "HN12345678",
    "vehicle_make": "Honda",
    "vehicle_model": "Accord",
    "vehicle_year": "2020",
    "incident_date": "2025-04-24T18:00:00",
    "incident_description": "My car was rear-ended while I was stopped at a red light.",
    "point_of_impact": "Rear"
    }

    Example 2:
    User: I got into a collision this morning. A truck hit the passenger side of my car 
    while I was merging. It’s a 2018 Toyota Camry.
    Output: {
    "vehicle_make": "Toyota",
    "vehicle_model": "Camry",
    "vehicle_year": "2018",
    "incident_date": "2025-04-25T08:00:00",
    "incident_description": "A truck hit the passenger side of my car while I was merging.",
    "point_of_impact": "Right Side"
    }

    Example 3:
    User: My name’s Angela, and my 2015 Ford Focus got t-boned on the driver side 
    last week Tuesday. The other driver ran a red light. 
    Output: {
    "policy_holder_name": "Angela",
    "vehicle_make": "Ford",
    "vehicle_model": "Focus",
    "vehicle_year": "2015",
    "incident_date": "2025-04-15T08:00:00",
    "incident_description": "My car got t-boned on the driver side. The other driver ran a red light.",
    "point_of_impact": "Left Side"
    }
    Example 4:
    There was a bad hailstorm last night and now the top of my BMW 3 Series 
    looks like it got hit with golf balls.
    Output: {
    "vehicle_make": "BMW",
    "vehicle_model": "3 Series",
    "incident_date": "2025-04-24T22:00:00",
    "incident_description": "A bad hailstorm caused damage to the top of my vehicle.",
    "point_of_impact": "Top"
    }
    Example 5:
    My Chevy Silverado 2021 got hit from the front and the right rear when I lost 
    control on an icy road. Policy is SILV998877.
    Output: {
    "policy_number": "SILV998877",
    "vehicle_make": "Chevrolet",
    "vehicle_model": "Silverado",
    "vehicle_year": "2021",
    "incident_description": "The vehicle got hit from the front and the right rear after losing control on an icy road.",
    "point_of_impact": "Multiple Points / Multiple Areas"
    }
    """,
    # instrument=True # Uncomment if using Logfire/LangSmith
)

sql_agent = PydanticAIAgent(
    LLM_MODEL,
    output_type=SQLResponse,
    system_prompt=f"""
    You are an expert SQLite query generator. Your task is to create a SQLite SELECT query 
    based on the user's request to retrieve information from the 'claims' table.

    **Database Schema:**
    ```sql
    {DB_SCHEMA}
    Use code with caution.
    Python
    Important Notes:
    The table name is "claims".
    Generate ONLY SELECT queries. Do NOT generate INSERT, UPDATE, DELETE, or other modifying queries.
    Use standard SQLite syntax. Pay attention to column names and types.
    The id column is the primary key (TEXT).
    incident_date is stored as DATETIME (ISO 8601 format string). Use functions like date(), datetime(), strftime() for date comparisons if needed. E.g., WHERE date(incident_date) = '2025-01-15'.
    Filter based on the details provided in the user's request (query_details).
    If the request is too vague or lacks specifics to form a query, respond using the InvalidSQLRequest schema.
    If the request seems valid, respond using the SQLQuery schema. Include a brief explanation if helpful.
    Examples:
    User Request (query_details): "claim ID CLM-9876543210"
    Output (SQLQuery): {{"sql": "SELECT * FROM claims WHERE id = 'CLM-9876543210';", "explanation": "Selects the claim matching the specified ID."}}
    User Request (query_details): "claims for policy holder John Doe"
    Output (SQLQuery): {{"sql": "SELECT * FROM claims WHERE policy_holder_name = 'John Doe';", "explanation": "Selects all claims for the policy holder named John Doe."}}
    User Request (query_details): "claims with status Approved for Alpha Insurance"
    Output (SQLQuery): {{"sql": "SELECT * FROM claims WHERE status = 'Approved' AND company = 'Alpha Insurance';", "explanation": "Selects approved claims from Alpha Insurance."}}
    User Request (query_details): "claims that happened yesterday"
    Output (SQLQuery): {{"sql": "SELECT * FROM claims WHERE date(incident_date) = date('now', '-1 day');", "explanation": "Selects claims where the incident occurred yesterday."}}
    User Request (query_details): "details about a claim"
    Output (InvalidSQLRequest): {{"error_message": "Please provide more specific details for the claim you want to retrieve, such as the claim ID or policy number."}}
    User Request (query_details): "delete claim 123"
    Output (InvalidSQLRequest): {{"error_message": "Sorry, I can only retrieve claim information. I cannot perform delete operations."}}
    """,
    instrument=True,  # Uncomment if using Logfire/LangSmith
)