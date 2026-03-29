# Shop4You — Submission Checklist

> **Learner:** Mohammed Emran
> **Date:** February 26, 2026
> **Project:** Agentic AI Assistant for Shop4You (Retail Company)

---

## Compulsory Goals

| # | Requirement | Evidence | Status |
|---|-------------|----------|--------|
| 1 | Finalise 4+ departments (≥2 internal, ≥2 external) | `config.py` — 8 departments (4 customer, 4 employee) | ✅ |
| 2 | Create QA datasets (10–15 QA per dept, LLM-generated) | `data/` — 8 JSON files, 25 pairs each = 200 total | ✅ |
| 3 | Create vector database with department metadata | `vector_store.py` + `knowledge_base/chroma/` — 200 vectors | ✅ |
| 4 | Build Agentic AI System (classify → route → RAG/escalate) | `agents.py` — LangGraph orchestrator + 7 agents (4 orchestration + 3 CrewAI) | ✅ |
| 5 | Test on sample queries | 13 pytest + 70 orchestration + 10 demo scenarios | ✅ |

## Stretch Goals (all 4 implemented)

| # | Stretch Goal | Evidence | Status |
|---|-------------|----------|--------|
| 1 | Multi-user conversational memory | `agents.py` — SqliteSaver, per-user threads | ✅ |
| 2 | Escalation with CrewAI investigation | `escalation_crew.py` — 3-agent sequential crew | ✅ |
| 3 | More departments + custom workflows | 8 depts, 6 tools (product, loyalty, orders, KB, depts, escalate) | ✅ |
| 4 | API / application deployment | `app.py` (FastAPI, 6 endpoints) + `streamlit_app.py` (Chat UI) | ✅ |

## Test Results Summary

| Test Suite | Count | Result |
|------------|-------|--------|
| Pytest (RAG + Agent + Memory) | 13 | 13/13 ✅ |
| Orchestration & Routing | 70 | 70/70 ✅ |
| Demo Showcase Scenarios | 10 | 10/10 ✅ |
| **Total** | **93** | **93/93 ✅** |

## How to Run

```powershell
conda activate shop4you
cd submission/

# Quick sanity check
python -c "from config import *; from agents import compile_graph; print('OK')"

# Run all 13 pytest tests
python -m pytest test_rag.py test_agent.py test_memory_isolation.py -v --tb=short

# Run 70 orchestration tests (~5 min)
python test_orchestration.py

# Launch Streamlit UI
streamlit run streamlit_app.py

# Launch FastAPI
python -m uvicorn app:app --reload

# Open presentation
# → docs/Shop4You_Presentation.html (open in browser, press F for fullscreen)
```

## File Inventory

### Core Code
- `agents.py` — LangGraph orchestrator (Agent 1) + compile/run helpers
- `config.py` — 8 department definitions, env vars, settings
- `prompts.py` — All ChatPromptTemplate instances
- `escalation_crew.py` — CrewAI 3-agent escalation crew (Agents 5, 6, 7)
- `vector_store.py` — ChromaDB ingestion + filtered retrieval
- `tools.py` — 6 @tool functions
- `users.py` — 10 demo users + auto-registration
- `orders_db.py` — 11 demo orders + lookup
- `generate_data.py` — Synthetic FAQ generator

### Interfaces
- `streamlit_app.py` — Streamlit Chat UI (login, chat, transfer, farewell)
- `app.py` — FastAPI REST API (6 endpoints)
- `main.py` — CLI with Rich

### Tests
- `conftest.py` — Shared pytest fixtures
- `test_rag.py` — 5 RAG pipeline tests
- `test_agent.py` — 7 E2E agent tests
- `test_memory_isolation.py` — Multi-user isolation test
- `test_orchestration.py` — 70 orchestration & routing tests
- `demo_customer_showcase.py` — 10-scenario Rich demo

### Data & Knowledge Base
- `data/` — 8 JSON FAQ files (200 QA pairs)
- `knowledge_base/chroma/` — ChromaDB persistent vectors (200)

### Documentation
- `README.md` / `README.html` — Project overview + architecture
- `docs/TESTING_GUIDE.md` — Step-by-step test instructions
- `docs/PROJECT_STEPS.md` / `.html` — 30-step project plan
- `docs/shop4you_data_reference.html` — Data reference + demo script
- `docs/project_brief.pdf` — Original capstone brief
- `docs/Shop4You_Presentation.html` — 35-slide Reveal.js presentation
- `docs/Shop4You_Architecture_Diagram.html` — Visual architecture diagram (colour-coded)

### Config
- `requirements.txt` — Pinned Python dependencies
- `environment.yml` — Conda environment spec
- `demo.ipynb` — Jupyter demo walkthrough
- `logo.svg` — Shop4You brand logo (used in Streamlit UI)
