# AutoStream Agentic Workflow

This is the solution repository for the ServiceHive ML Intern assignment: Social-to-Lead Agentic Workflow. 

## 1. How to run the project locally

1. **Prerequisites**: Python 3.10+
2. **Setup Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Configure API Key**:
   - Copy `.env.example` to `.env`.
   - Get a free free API Key from [Google AI Studio](https://aistudio.google.com/).
   - Add it to `.env`: `GOOGLE_API_KEY="your_actual_key"`
4. **Run Agent**:
   ```bash
   python main.py
   ```
 *(Note: The knowledge base vector index is built automatically on the first run using `sentence-transformers` locally, so it may take a few seconds to initialize).*

---

## 2. Architecture Explanation

This agent leverages **LangGraph** paired with **Google Gemini 1.5 Flash**. 

**Why LangGraph?**  
LangGraph forces predictable state machine structures, which is ideal for multi-step agentic workflows. Instead of relying on a completely autonomous ReAct loop (which can hallucinate API calls or prematurely trigger lead capture), LangGraph allows us to build distinct 'lanes': one for knowledge retrieval and an explicit conditional branch for multi-turn lead collection. 

**State Management:**  
State carries across multi-turn interactions using LangGraph's `MemorySaver` bound to a unique `thread_id`. The custom `AgentState` (`TypedDict`) maintains conversation `messages`, calculated `intent`, a `lead_info` dictionary, and a `lead_captured` flag. 

**Workflow Nodes:**  
1. **`intent_router`**: Uses structured LLM output to classify the latest message into *greeting, inquiry,* or *high_intent*. 
2. **`rag_responder`**: Retrieves context from local FAISS (embedded via `all-MiniLM-L6-v2`) and answers product questions.  
3. **`lead_collector`**: Uses structured extraction to continuously parse missing `name`, `email`, and `platform` from conversation history. If any are missing, it responds proactively. Once all three exist in state, it safely executes the `mock_lead_capture` Python tool and flips `lead_captured` to True.

---

## 3. WhatsApp Deployment Question

> **Q:** Explain how you would integrate this agent with WhatsApp using Webhooks:

To integrate this agent with the official WhatsApp Business API via webhooks, I would use a lightweight API framework like **FastAPI** to wrap the LangGraph backend.

1. **Verify Webhook**: Build a `GET` endpoint confirming the Hub Challenge token from Meta so WhatsApp can validate my server.
2. **Receive Messages**: Expose a `POST` webhook to receive JSON payloads whenever users send a message. I'd extract the user's phone number and message contents.
3. **State Persistance via Phone Number**: Instead of a random `uuid` for the LangGraph `thread_id` (as in `main.py`), I'd use the user's phone number as the persistent `thread_id`. The current `MemorySaver` could be swapped for `langgraph-checkpoint-postgres` to keep state persistent across server restarts natively.
4. **Agent Processing Background**: Pass the user's WhatsApp message to the LangGraph `graph.invoke()` asynchronously so Meta's standard 3-second webhook timeout isn't breached. 
5. **Reply API Call**: Once the LangGraph returns its response, format the output as a WhatsApp Cloud API JSON structure and execute an outbound HTTP POST to `https://graph.facebook.com/v19.0/.../messages` to send the response back.
