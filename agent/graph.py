from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agent.state import AgentState
from agent.nodes import intent_router_node, rag_responder_node, lead_collector_node

def route_next_node(state: AgentState) -> str:
    """Conditional edge logic based on intent."""
    intent = state.get("intent")
    
    if intent == "high_intent":
        return "lead_collector"
    else:
        # "greeting" or "inquiry" both go to RAG responder 
        # (the rag node handles non-RAG simple greetings internally too)
        return "rag_responder"

# Build the Graph Builder
builder = StateGraph(AgentState)

# Add Nodes
builder.add_node("intent_router", intent_router_node)
builder.add_node("rag_responder", rag_responder_node)
builder.add_node("lead_collector", lead_collector_node)

# Add Edges
builder.set_entry_point("intent_router")
builder.add_conditional_edges("intent_router", route_next_node)

# Final step: everything ends after generating an agent response
builder.add_edge("rag_responder", END)
builder.add_edge("lead_collector", END)

# Compile into an executable LangChain Runnable
# We use MemorySaver to persist state across conversation turns
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
