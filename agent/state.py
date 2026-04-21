import operator
from typing import Annotated, TypedDict

class AgentState(TypedDict):
    """
    State for the AutoStream LangGraph Agent.
    """
    # Using operator.add to append new messages to the existing list
    messages: Annotated[list, operator.add]
    
    # State tracking for intent
    intent: str
    
    # Lead capture details
    lead_name: str | None
    lead_email: str | None
    lead_platform: str | None
    
    # Flags
    lead_captured: bool
    
