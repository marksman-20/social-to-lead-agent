import json
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agent.state import AgentState
from agent.prompts import (
    SYSTEM_PROMPT,
    INTENT_ROUTER_PROMPT,
    RAG_ANSWER_PROMPT,
    LEAD_EXTRACTION_PROMPT
)
from agent.tools import mock_lead_capture
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from knowledge.rag_retriever import get_retriever

# Initialize LLM & Retriever
# Using gemini-1.5-flash as per requirements
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
retriever = get_retriever()

class IntentClassification(BaseModel):
    intent: str = Field(description="Must be one of: 'greeting', 'inquiry', 'high_intent'")

class LeadExtraction(BaseModel):
    lead_name: str | None = Field(default=None)
    lead_email: str | None = Field(default=None)
    lead_platform: str | None = Field(default=None)


def intent_router_node(state: AgentState) -> dict:
    """Classifies user intent to route to the appropriate node."""
    messages = state["messages"]
    last_message = messages[-1].content

    # If lead capture was already complete, reset to greeting so we don't loop endlessly
    if state.get("lead_captured", False):
        # We could also use this step to handle follow-up questions
        # For simplicity, we just route everything else to rag/greeting
        pass

    prompt = INTENT_ROUTER_PROMPT.format(message=last_message)
    structured_llm = llm.with_structured_output(IntentClassification)
    
    try:
        response = structured_llm.invoke(prompt)
        intent = response.intent
    except Exception as e:
        # Fallback if structure parsing fails
        print(f"[DEBUG] Intent fallback logic hit: {e}")
        intent = "inquiry"
        
    # If the user is already stuck in high_intent and providing lead details, keep them there!
    # A name like "John" might get classified as "greeting" or "inquiry" otherwise.
    if state.get("intent") == "high_intent" and not state.get("lead_captured", False):
        intent = "high_intent"

    return {"intent": intent}


def rag_responder_node(state: AgentState) -> dict:
    """Handles greetings and product inquiries using RAG."""
    messages = state["messages"]
    last_message = messages[-1].content
    intent = state.get("intent", "greeting")
    
    if intent == "greeting":
        # Simple AI response without RAG
        prompt = [
            SystemMessage(content=SYSTEM_PROMPT),
            messages[-1] 
        ]
        response = llm.invoke(prompt)
        return {"messages": [response]}
        
    # Else, it's an inquiry - Use RAG
    docs = retriever.invoke(last_message)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    prompt_str = RAG_ANSWER_PROMPT.format(context=context, question=last_message)
    
    prompt = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt_str)
    ]
    response = llm.invoke(prompt)
    
    return {"messages": [response]}


def lead_collector_node(state: AgentState) -> dict:
    """Handles high-intent users, asks for info, and calls mock API."""
    messages = state["messages"]
    
    # 1. Extract any new info from the latest user message
    # We pass the last 3-4 messages for context
    history_str = ""
    for msg in messages[-4:]:
        role = "User" if isinstance(msg, HumanMessage) else "Agent"
        history_str += f"{role}: {msg.content}\n"
        
    extractor = llm.with_structured_output(LeadExtraction)
    try:
        extracted = extractor.invoke(LEAD_EXTRACTION_PROMPT.format(history=history_str))
        
        # Merge with existing state
        new_name = extracted.lead_name or state.get("lead_name")
        new_email = extracted.lead_email or state.get("lead_email")
        new_platform = extracted.lead_platform or state.get("lead_platform")
    except Exception as e:
        print(f"[DEBUG] Extraction fallback logic hit: {e}")
        new_name = state.get("lead_name")
        new_email = state.get("lead_email")
        new_platform = state.get("lead_platform")

    updates = {
        "lead_name": new_name,
        "lead_email": new_email,
        "lead_platform": new_platform
    }
    
    # 2. Check what's missing
    missing = []
    if not new_name:
        missing.append("name")
    if not new_email:
        missing.append("email address")
    if not new_platform:
        missing.append("creator platform (e.g. YouTube, Instagram)")
        
    # 3. Determine agent response
    if missing:
        # Prompt for the first missing thing
        asking_for = missing[0]
        prefix = ""
        # If this is the very first time we hit lead collection, acknowledge their intent
        if not state.get("lead_name") and not state.get("lead_email") and not state.get("lead_platform"):
           prefix = "That's great to hear! Let's get you set up. " 
            
        ai_msg = AIMessage(content=f"{prefix}Could you please provide your {asking_for}?")
        updates["messages"] = [ai_msg]
    else:
        # We have everything! Call the tool.
        mock_lead_capture(new_name, new_email, new_platform)
        
        ai_msg = AIMessage(content=f"Thanks, {new_name}! I have successfully registered your interest for {new_platform}. Our team will contact you at {new_email} shortly.")
        updates["messages"] = [ai_msg]
        updates["lead_captured"] = True
        
    return updates
