# Shop4You Capstone — Complete Project Steps

> **Capstone Project:** Agentic AI Assistant for Shop4You (Retail Company)  
> **Learner:** Mohammed Emran  
> **Updated:** February 23, 2026  
> **Stretch Goals:** All 4 selected (Multi-user memory, Escalation form, More departments, FastAPI deploy)

---

## PHASE 1: Setup & Data (Week 1)

### Step 1 — Create Project Directory Structure
- Create folders: `data/`, `knowledge_base/`, `agents/`, `tools/`, `config/`, `api/`
- Initialize virtual environment and `.gitignore`

### Step 2 — Finalize Departments (8+ total)
- **External (customer-facing):**
  1. Orders & Returns
  2. Billing / Payment
  3. Shipping / Delivery
  4. Product Inquiries *(Stretch Goal 3)*
- **Internal (employee-facing):**
  5. HR
  6. IT Helpdesk
  7. Operations
  8. Loyalty Program / Employee Benefits *(Stretch Goal 3)*
- Define each department's scope, audience, and tone

### Step 3 — Generate Synthetic QA Datasets
- Use an LLM to generate **10–15 QA pairs per department** (80–120 total)
- Prompt must specify: department role, internal vs external audience, tone, number of pairs
- Each entry: `question`, `answer`, `tags`, `last_updated`, `confidence_score`

### Step 4 — Store Datasets as JSON
- One JSON file per department
- Schema: `{ "department": "...", "doc_id": "...", "entries": [ { "question": "...", "answer": "...", "tags": [...] } ] }`

### Step 5 — Set Up Environment Variables
- Create `.env` with `OPENAI_API_KEY`, `TAVILY_API_KEY`
- Use `python-dotenv` for loading

### Step 6 — Install All Dependencies
```
langchain, langchain-openai, langchain-community, langgraph
langchain-chroma, langgraph-checkpoint-sqlite
fastapi, uvicorn, pydantic
python-dotenv, rich, tiktoken
```

### Step 7 — Create Embeddings & Ingest into ChromaDB
- Embed all QA docs using `OpenAIEmbeddings(model='text-embedding-3-small')`
- Store in ChromaDB with `department` metadata for filtered retrieval
- Persist to `./knowledge_base`

---

## PHASE 2: Core RAG Pipeline (Week 2)

### Step 8 — Build Retriever with Metadata Filtering
- `kbase_db.as_retriever()` with `similarity_score_threshold`
- Dynamic `filter={"department": dept}` per query

### Step 9 — Create RAG Prompt Templates
- Per-department prompt templates using `ChatPromptTemplate`
- Include customer query + retrieved context
- Fallback: "Apologies, I was not able to answer your question..."

### Step 10 — Build RAG Worker Node
- Input: department + user query
- Process: retrieve context → inject into prompt → LLM generates answer
- Output: formatted response with citations

### Step 11 — Test RAG Independently
- Verify each department retrieves correct documents
- Test edge cases: ambiguous queries, empty retrieval, cross-department queries

---

## PHASE 3: Router & Agentic Logic (Week 3)

### Step 12 — Build Router Agent
- LLM classifies **department** using `with_structured_output(QueryCategory)`
- LLM classifies **sentiment** using `with_structured_output(QuerySentiment)`
- Pydantic models:
  ```python
  class QueryCategory(BaseModel):
      department: Literal['Orders', 'Billing', 'Shipping', 'Product', 'HR', 'IT', 'Operations', 'Loyalty', 'Unknown']
  
  class QuerySentiment(BaseModel):
      sentiment: Literal['Positive', 'Neutral', 'Negative']
  ```

### Step 13 — Build Routing Logic
- `add_conditional_edges()` from router node:
  - **Negative sentiment OR Unknown department** → `escalate`
  - **Valid department + Positive/Neutral** → `department_rag`

### Step 14 — Build Escalation Node with Form (Stretch Goal 2)
- Collect user details: **name, phone number, email**
- Display confirmation: "Your info has been received, someone will reach out soon"
- Simulate sending email/WhatsApp notification
- Store escalation info in state: `escalation_info` dict

### Step 15 — Build Reflection / Self-Check Loop
- After generating RAG answer, run quality checks:
  - Retrieval relevance score
  - Hallucination risk assessment
  - Policy compliance check
- If quality low → retry (up to `MAX_ATTEMPTS = 3`)
- If quality OK → pass to response

### Step 16 — Build Custom Workflows for Extra Departments (Stretch Goal 3)
- **Product Inquiries:** product search tool, stock availability check
- **Loyalty Program:** points balance lookup, rewards info
- Specialized tools per department using `@tool` decorator

### Step 17 — Add Multi-User Conversational Memory (Stretch Goal 1)
- Use `SqliteSaver.from_conn_string("shop4you_memory.db")`
- Each user gets unique `thread_id`: `config = {"configurable": {"thread_id": f"user_{user_id}"}}`
- Conversations persist across sessions
- Users don't see each other's history

### Step 18 — Define State Schema
```python
class Shop4YouState(TypedDict):
    messages: Annotated[list, add_messages]
    customer_query: str
    department: str
    sentiment: str
    retrieved_context: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    reflection_count: int
    final_response: str
    user_id: str
    escalation_info: dict
```

### Step 19 — Wire the Full LangGraph
```
START
  → Step 0: classify_query (department, sentiment, severity)
  → route_query (conditional — one path only)
      ├─ IF valid dept (or low/med negative):
      │    → Step 1a: department_rag (ChromaDB + tools + GPT-4o)
      │    → Step 2:  reflect (quality score ≥7 → done, else retry up to 3×)
      │    → END
      └─ IF negative+HIGH severity OR unknown dept:
           → Step 1b: escalate (CrewAI 3-agent crew → escalation_agent)
           → END
```

> **📊 Visual version:** Open [`docs/Shop4You_Architecture_Diagram.html`](Shop4You_Architecture_Diagram.html) for a colour-coded interactive diagram.

---

## PHASE 4: API & Deployment (Week 4 — First Half) — Stretch Goal 4

### Step 20 — Create FastAPI App
- `POST /chat` endpoint: accepts `{ "query": "...", "user_id": "..." }`
- Returns `{ "response": "...", "department": "...", "sentiment": "..." }`
- Swagger UI at `/docs`

### Step 21 — Integrate Agent with API
- API calls the compiled LangGraph agent
- Passes `user_id` as `thread_id` for memory
- Returns structured response

### Step 22 — Add Session Management
- API tracks user sessions via `thread_id`
- Support `GET /history/{user_id}` to retrieve conversation history
- Support `DELETE /history/{user_id}` to clear a user's session

### Step 23 — Deploy the Web Service
- Run with `uvicorn api.main:app --reload`
- Test via Swagger UI, curl, or Postman
- Optional: deploy to cloud (Render, Railway, or AWS)

---

## PHASE 5: Testing & Polish (Week 4 — Second Half)

### Step 24 — End-to-End Testing
Test 10+ sample queries covering:
- [x] Each department (8+ queries, one per dept)
- [x] Negative sentiment → escalation with form collection
- [x] Unknown department → escalation
- [x] Multi-turn conversations (same user returns, context recalled)
- [x] Different users don't see each other's history
- [x] API endpoint testing via Swagger/Postman
- [x] Reflection loop triggers on low-quality retrieval
- [x] **Severity-based escalation** (only HIGH severity escalates, not all negative)
- [x] **Cross-department handoff** (SUGGEST_TRANSFER tag + fallback detection)
- [x] **Transfer keyword detection** (22 phrases including "pass me to", "connect me")
- [x] **Generic term handling** ("customer service" → contextual department)
- [x] **Conversation-aware classification** (last 3 turns for follow-up context)
- [x] **No "contact customer service" in responses** (KB data fixed + RAG override)
- [x] **70 orchestration tests passed** across 12 categories
- [x] **13 pytest tests passed** (5 RAG + 7 Agent + 1 Memory Isolation) via `conftest.py` shared fixtures
- [x] **10/10 demo scenarios** scored 9-10/10 in `demo_customer_showcase.py`
- [x] **Reflection prompt improved** — recognises tool results and memory as legitimate data (not hallucination)

### Step 25 — Add Logging and Metrics
- Query logs with timestamps
- Retrieval precision: fraction of relevant passages
- Escalation rate: percent of queries routed to human
- Response latency tracking

### Step 26 — Verify Memory Isolation
- User A and User B have independent conversations
- User A can say "what did we discuss" and get only their history
- Restarting the app preserves conversation history (SQLite persistence)

---

## PHASE 6: Deliverables & Presentation

### Step 27 — Prepare Demo Script
- 10+ sample queries with expected outputs
- Cover all departments, escalation paths, multi-turn, and API
- Include edge cases and error scenarios

### Step 28 — Create Presentation / Slide Deck
- Architecture diagram (LangGraph flow) — see also [`docs/Shop4You_Architecture_Diagram.html`](Shop4You_Architecture_Diagram.html)
- Demo walkthrough screenshots
- Key patterns used (routing, RAG, reflection, planning, multi-agent)
- Business impact and scalability

### Step 29 — Write README and Documentation
- `README.md` — project overview, quick start
- `SETUP_GUIDE.md` — step-by-step installation
- `API_REFERENCE.md` — FastAPI endpoints documentation
- `PROJECT_DOCUMENTATION.md` — complete technical guide for evaluators

### Step 30 — Final Mentor Walkthrough and Submission
- Meet with mentor to review progress
- Do a live project demo
- Submit all code, data, documentation, and presentation

---

## Summary

| Phase | Steps | Duration |
|-------|-------|----------|
| Phase 1: Setup & Data | Steps 1–7 | Week 1 |
| Phase 2: Core RAG Pipeline | Steps 8–11 | Week 2 |
| Phase 3: Router & Agentic Logic | Steps 12–19 | Week 3 |
| Phase 4: API & Deployment | Steps 20–23 | Week 4 (first half) |
| Phase 5: Testing & Polish | Steps 24–26 | Week 4 (second half) |
| Phase 6: Deliverables | Steps 27–30 | Final days |

**Total: 30 steps | 6 phases | 4 stretch goals included**
