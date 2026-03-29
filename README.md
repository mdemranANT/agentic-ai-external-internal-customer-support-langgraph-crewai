# Shop4You Capstone Project - Agentic AI Assistant

> Learner: Mohammed Emran  
> Project Type: Multi-agent customer and employee support system  
> Domain: Retail (Shop4You)

## 1. Project Objective
Build an end-to-end Agentic AI assistant that can classify user queries, route to the right department, retrieve grounded answers from a vector database, and escalate critical cases with a multi-agent investigation workflow.

## 2. Assignment Alignment

### 2.1 Compulsory Goals

| Requirement | Implementation in this project | Status |
|---|---|---|
| Finalize 4+ departments (internal + external) | 8 departments in `config.py` (4 customer-facing, 4 employee-facing) | Completed |
| Create synthetic QA datasets | 8 JSON files in `data/`, 25 pairs each (200 total) | Completed |
| Build vector database with metadata | ChromaDB ingestion in `vector_store.py`, department metadata filtering | Completed |
| Build Agentic routing system | LangGraph workflow in `agents.py` (classify -> route -> RAG/escalation -> reflection) | Completed |
| Test on sample queries | Pytest suites + orchestration tests + demo showcase | Completed |

### 2.2 Stretch Goals

| Stretch Goal | Implementation | Status |
|---|---|---|
| Multi-user conversational memory | SQLite checkpointer with per-user thread IDs | Completed |
| Escalation workflow | CrewAI 3-agent escalation pipeline in `escalation_crew.py` | Completed |
| Additional departments/custom tools | 8 departments + 6 tools | Completed |
| API / app deployment | FastAPI service (`app.py`) + Streamlit interface (`streamlit_app.py`) | Completed |

## 3. End-to-End Workflow

1. User asks a query from CLI, Streamlit, or API.
2. Classifier identifies department, sentiment, and severity.
3. Router sends query to:
   - Department RAG path (normal flow), or
   - Escalation path (high-severity negative or unknown cases).
4. RAG path retrieves relevant context from ChromaDB (department-filtered) and can call tools.
5. LLM generates response with context + tool outputs.
6. Reflection node scores quality and retries when needed (bounded loop).
7. Multi-user memory persists conversation context per user.

## 4. Agents and Tools

### 4.1 Agents
- Orchestration agents (LangGraph): Classifier, Department RAG, Reflection, Escalation
- Escalation crew agents (CrewAI): Complaint Analyst, Account Investigator, Resolution Specialist

### 4.2 Tools
- Knowledge base search
- Department listing
- Product lookup
- Loyalty lookup
- Order lookup
- Human escalation trigger

## 5. Project Structure

```text
submission/
  agents.py
  app.py
  config.py
  escalation_crew.py
  vector_store.py
  tools.py
  users.py
  orders_db.py
  streamlit_app.py
  main.py
  test_rag.py
  test_agent.py
  test_memory_isolation.py
  test_orchestration.py
  demo_customer_showcase.py
  demo.ipynb
  data/
  docs/
```

## 6. Setup and Run

### 6.1 Environment Setup

```powershell
conda env create -f environment.yml
conda activate shop4you
```

### 6.2 Configure Secrets
Create `.env` in `submission/` with:

```env
OPENAI_API_KEY=your_key_here
TAVILY_API_KEY=optional_key_here
```

### 6.3 Build/Refresh Vector Store

```powershell
python vector_store.py
```

### 6.4 Run Interfaces

```powershell
# CLI
python main.py

# Streamlit
streamlit run streamlit_app.py

# FastAPI
python -m uvicorn app:app --reload
```

## 7. Testing Summary

### 7.1 Core Tests

```powershell
python -m pytest test_rag.py test_agent.py test_memory_isolation.py -v --tb=short
```

### 7.2 Orchestration Suite

```powershell
python test_orchestration.py
```

### 7.3 Showcase Demo

```powershell
python demo_customer_showcase.py
```

## 8. Key Capabilities Demonstrated

- Department-aware routing for customer and employee support.
- Retrieval-augmented response generation with metadata filtering.
- Severity-aware escalation (not all negative sentiment escalates).
- Cross-department handling and transfer support.
- Per-user memory isolation and persistence.
- API + UI deployment-ready architecture.

## 9. Documentation

- `docs/PROJECT_STEPS.md` - full phase-wise implementation plan
- `docs/TESTING_GUIDE.md` - complete testing walkthrough
- `docs/Shop4You_Architecture_Diagram.html` - visual architecture
- `docs/Shop4You_Presentation.html` - project presentation deck
- `docs/shop4you_data_reference.html` - data and scenario reference

## 10. Submission Notes

This repository version is focused on the final deliverable implementation in the `submission/` folder.
All components are aligned to capstone requirements and include both compulsory and stretch-goal coverage.
