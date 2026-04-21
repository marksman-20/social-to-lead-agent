import os
import uuid
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# Load environment variables (API Key) before initializing any LangChain tools
load_dotenv()

# Basic validation
if not os.environ.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY") == "your_api_key_here":
    print("⚠️ WARNING: GOOGLE_API_KEY is not set or is using the default template.")
    print("Please update '.env' with a valid Gemini API Key from https://aistudio.google.com/")
    print("Exiting...")
    exit(1)

from agent.graph import graph

def print_banner():
    print("\n" + "="*50)
    print(" Welcome to the AutoStream AI Assistant!")
    print(" Type 'exit' or 'quit' to end the conversation.")
    print("="*50 + "\n")

def main():
    print_banner()
    
    # We use a distinct thread_id for this conversation session.
    # The MemorySaver checkpointer uses this thread_id to persist states.
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Session ended. Goodbye!")
            break
            
        if not user_input.strip():
            continue
            
        # Stream the graph execution
        inputs = {"messages": [HumanMessage(content=user_input)]}
        
        # Invoke graph and stream output
        # .stream() lets us watch intermediate nodes if we want, but we just want the final AI response
        print("Agent: ", end="", flush=True)
        try:
            for event in graph.stream(inputs, config, stream_mode="values"):
                # stream_mode="values" returns the full state after each update.
                pass
            
            # Print the final message content
            final_state = graph.get_state(config).values
            print(final_state["messages"][-1].content)
            
        except Exception as e:
            print(f"[Error] Graph execution failed: {e}")

if __name__ == "__main__":
    main()
