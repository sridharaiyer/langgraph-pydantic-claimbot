from datetime import datetime
from typing import Optional, Literal

from pydantic import BaseModel, Field


class Intent(BaseModel):
    """Model to hold user intent and query details."""
    action: Literal["create", "retrieve", "unknown"] = Field(
        description="The user's intent: create a new claim, retrieve existing claims, or unknown.")
    query_details: Optional[str] = Field(
        None, description="Specific details mentioned by the user for retrieval (e.g., 'claim ID CLM-123', 'claims for John Doe', 'claims with status Approved'). Null if the intent is 'create' or 'unknown'.")


class PartialClaim(BaseModel):
    """Model to hold extracted claim details, all optional."""
    policy_holder_name: Optional[str] = Field(None, title="Policy Holder Name")
    policy_number: Optional[str] = Field(None, title="Policy Number")
    vehicle_make: Optional[str] = Field(None, title="Vehicle Make")
    vehicle_model: Optional[str] = Field(None, title="Vehicle Model")
    vehicle_year: Optional[int] = Field(None, title="Vehicle Year")
    incident_date: Optional[datetime] = Field(None, title="Incident Date")
    incident_description: Optional[str] = Field(
        None, title="Incident Description")
    adjuster_name: Optional[str] = Field(None, title="Adjuster Name")
    status: Optional[str] = Field(None, title="Status")
    company: Optional[str] = Field(None, title="Company")
    claim_office: Optional[str] = Field(None, title="Claim Office")
    point_of_impact: Optional[str] = Field(None, title="Point Of Impact")
