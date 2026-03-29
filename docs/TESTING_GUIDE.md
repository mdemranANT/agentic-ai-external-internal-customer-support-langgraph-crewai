# Shop4You — Complete Testing Guide

> Run every verification step yourself and demonstrate the project to your instructor.  
> **Estimated total time:** ~5 minutes (excluding API server demo).

---

## Prerequisites

| Requirement | How to check |
|---|---|
| Conda environment **shop4you** | `conda env list` — you should see `shop4you` in the list |
| `.env` file with `OPENAI_API_KEY` | `type .env` — must contain your key |
| Knowledge base already ingested | Folder `knowledge_base/chroma/` should exist |

**Activate the environment first (every new terminal):**

```powershell
conda activate shop4you
cd ..
```

---

## Step 0 — Quick Sanity Check (Imports + Config)

Verifies all modules load, the API key works, and CrewAI is enabled.

```powershell
python -c "
from config import DEPARTMENTS, OPENAI_API_KEY, CREWAI_ESCALATION_ENABLED
from vector_store import create_vector_store, retrieve_context
from agents import compile_graph, run_query
from tools import ALL_TOOLS
print('All imports .............. OK')
print(f'Departments .............. {len(DEPARTMENTS)}')
print(f'Tools .................... {len(ALL_TOOLS)}')
print(f'CrewAI escalation ........ {CREWAI_ESCALATION_ENABLED}')
print(f'API key loaded ........... {bool(OPENAI_API_KEY)}')
"
```

**Expected output:**
```
All imports .............. OK
Departments .............. 8
Tools .................... 6
CrewAI escalation ........ True
API key loaded ........... True
```

---

## Step 1 — Data Integrity Check

Confirms 200 QA pairs (8 departments × 25) and 200 ChromaDB vectors.

```powershell
python -c "
import json, os, chromadb

total = 0
for f in sorted(os.listdir('data')):
    if f.endswith('.json'):
        with open(f'data/{f}') as fh:
            data = json.load(fh)
            count = len(data.get('entries', data)) if isinstance(data, dict) else len(data)
            total += count
            print(f'  {f}: {count} pairs')
print(f'\nTotal QA pairs: {total}')

client = chromadb.PersistentClient(path='./knowledge_base/chroma')
cols = client.list_collections()
for c in cols:
    print(f'ChromaDB [{c.name}]: {c.count()} vectors')
"
```

**Expected output:** Every JSON file shows **25 pairs**, total **200**, ChromaDB collection `shop4you_kb` has **200 vectors**.

---

## Step 2 — RAG Pipeline Tests (5 tests)

Tests retrieval accuracy per department, fallback handling, generation quality, edge cases, and metadata isolation.

```powershell
# Standalone (Rich-formatted output)
python test_rag.py

# Or via pytest (uses shared fixtures from conftest.py)
python -m pytest test_rag.py -v --tb=short
```

**Expected output:** A rich table showing **5/5 tests passed** (standalone) or **5 passed** (pytest).

### What each test checks:
| # | Test | What it verifies |
|---|---|---|
| 1 | Retrieval per department | A query for each of 8 departments returns relevant documents from the correct department |
| 2 | Fallback retrieval | A gibberish query still returns something (graceful degradation) |
| 3 | RAG generation | The LLM produces a coherent answer using retrieved context |
| 4 | Edge cases | Ambiguous or adversarial queries don't crash the system |
| 5 | Metadata isolation | A billing query does NOT pull documents from HR, etc. |

---

## Step 3 — Full Agent End-to-End Tests (7 tests)

Tests the complete LangGraph agent: routing, RAG, escalation (with CrewAI), product tools, and multi-turn memory.

```powershell
# Standalone (Rich-formatted output)
python test_agent.py

# Or via pytest (uses shared fixtures from conftest.py)
python -m pytest test_agent.py -v --tb=short
```

**Expected output:** A rich table showing **7/7 tests passed** (standalone) or **7 passed** (pytest).

### What each test checks:
| # | Test | What it verifies |
|---|---|---|
| 1 | Normal RAG (billing) | Billing query routes to `billing_payments`, gets a quality answer |
| 2 | Normal RAG (HR) | HR query routes to `hr`, gets a quality answer |
| 3 | Product search tool | Product query triggers the product search tool (Stretch Goal 3) |
| 4 | Negative sentiment → Escalation | Angry message triggers escalation + CrewAI investigation crew |
| 5 | Unknown dept → Escalation | Off-topic query triggers escalation + CrewAI investigation crew |
| 6 | Multi-turn memory (turn 1) | User introduces themselves and asks about an order |
| 7 | Multi-turn memory (turn 2) | Follow-up query recalls info from turn 1 (same user_id) |

> **Note:** Escalation tests take ~15–20 seconds each because the CrewAI 3-agent crew runs sequentially.

---

## Step 4 — Memory Isolation Test

Proves that separate user sessions don't leak into each other.

```powershell
# Standalone (Rich-formatted output)
python test_memory_isolation.py

# Or via pytest (uses shared fixtures from conftest.py)
python -m pytest test_memory_isolation.py -v --tb=short
```

**Expected output:**
```
User A dept: ...  sentiment: ...
User B dept: ...  sentiment: ...
User A follow-up mentions Sarah: True
User B follow-up does NOT mention Sarah: True
User B follow-up mentions James: True

MEMORY ISOLATION: PASS
```

**What it does:**
1. User A says "My name is **Sarah**…" and asks about a refund.
2. User B says "My name is **James**…" and asks about a payslip.
3. User A's follow-up correctly recalls "Sarah".
4. User B's follow-up correctly recalls "James" and does **NOT** mention "Sarah".

---

## Step 5 — FastAPI Server + API Tests

### 5a. Start the server

First, clear any stale memory database, then start the server:

```powershell
# Remove old memory DB to avoid stale ChromaDB references
Remove-Item shop4you_memory.db -Force -ErrorAction SilentlyContinue

# Start the server
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

The terminal will show:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
```

> **Important:** Keep this terminal open. Open a **new terminal** for the API tests below.

### 5b. Open Swagger UI (for instructor demo)

Open your browser and go to: **http://127.0.0.1:8000/docs**

This gives you an interactive UI where you can test every endpoint live.

### 5c. Run API tests from a second terminal

Open a **new PowerShell terminal**, activate conda, and run:

```powershell
conda activate shop4you
cd ..
```

Then test each endpoint:

#### Health check
```powershell
python -c "
import requests
r = requests.get('http://127.0.0.1:8000/health')
print(r.json())
"
```
**Expected:** `{'status': 'healthy', 'service': 'Shop4You AI Assistant', 'version': '1.0.0'}`

#### List departments
```powershell
python -c "
import requests
r = requests.get('http://127.0.0.1:8000/departments')
depts = r.json()
print(f'{len(depts)} departments:')
for d in depts:
    print(f'  {d[\"key\"]}: {d[\"name\"]} ({d[\"audience\"]})')
"
```
**Expected:** 8 departments listed with keys, names, and audience types.

#### Chat — Normal RAG query
```powershell
python -c "
import requests, json
r = requests.post('http://127.0.0.1:8000/chat', json={'query': 'What payment methods do you accept?', 'user_id': 'demo_user'})
data = r.json()
print(f'Department: {data[\"department\"]}')
print(f'Sentiment:  {data[\"sentiment\"]}')
print(f'Quality:    {data[\"quality_score\"]}')
print(f'Escalated:  {data[\"escalated\"]}')
print(f'Response:   {data[\"response\"][:200]}...')
"
```
**Expected:** `department: billing_payments`, `escalated: False`, quality ≥ 7.

#### Chat — Escalation with CrewAI
```powershell
python -c "
import requests
r = requests.post('http://127.0.0.1:8000/chat', json={'query': 'This is unacceptable! I have been waiting 3 weeks and nobody cares!', 'user_id': 'angry_user'})
data = r.json()
print(f'Escalated:  {data[\"escalated\"]}')
print(f'Reference:  {data[\"reference_number\"]}')
print(f'Department: {data[\"department\"]}')
print(f'Response:   {data[\"response\"][:300]}...')
"
```
**Expected:** `escalated: True`, a reference number like `ESC-XXXXX`, and the response includes a CrewAI investigation report.

#### Chat — Cross-department transfer
```powershell
python -c "
import requests, json
# First, send a query that may suggest a transfer
r1 = requests.post('http://127.0.0.1:8000/chat', json={'query': 'I want to return a jacket I bought', 'user_id': 'transfer_test'})
data1 = r1.json()
print(f'Department: {data1[\"department\"]}')
print(f'Severity:   {data1[\"severity\"]}')
print(f'Transfer:   {data1.get(\"suggested_transfer\", \"-\")}')
print()

# Transfer endpoint — force a query to a specific department
r2 = requests.post('http://127.0.0.1:8000/transfer', json={'query': 'What is your return policy?', 'target_department': 'orders_returns', 'user_id': 'transfer_test'})
data2 = r2.json()
print(f'Transfer dept: {data2[\"department\"]}')
print(f'Escalated:     {data2[\"escalated\"]}')
print(f'Response:      {data2[\"response\"][:200]}...')
"
```
**Expected:** First call returns `severity` field; transfer call routes to `orders_returns` with `escalated: False`.

#### Conversation history
```powershell
python -c "
import requests
r = requests.get('http://127.0.0.1:8000/history/demo_user')
data = r.json()
print(f'Messages: {data[\"message_count\"]}')
for m in data['messages']:
    print(f'  [{m[\"role\"]}] {m[\"content\"][:80]}...')
"
```
**Expected:** Shows previous messages for `demo_user`.

#### Clear history
```powershell
python -c "
import requests
r = requests.delete('http://127.0.0.1:8000/history/demo_user')
print(r.json())
"
```
**Expected:** `{'user_id': 'demo_user', 'status': 'cleared', ...}`

### 5d. Stop the server

Go back to the terminal running uvicorn and press **Ctrl+C**.

---

## Step 6 — Interactive Notebook Demo (demo.ipynb)

This is the best way to **show your instructor** the project interactively.

1. Open `demo.ipynb` in VS Code or Jupyter.
2. Make sure the kernel is set to the **shop4you** conda environment.
3. **Run all cells top to bottom** (Shift+Enter through each cell, or "Run All").

### What the notebook demonstrates (cell by cell):

| Cell | What it shows |
|---|---|
| 1 | Imports + compiles the LangGraph agent |
| 2 | Defines a `show()` helper to display results nicely |
| 3 | **Billing query** — normal RAG, high quality score |
| 4 | **Product inquiry** — triggers product search tool |
| 5 | **Loyalty program** — normal RAG |
| 6 | **Negative sentiment** — triggers escalation + CrewAI 3-agent investigation |
| 7 | **Unknown department** — triggers escalation + CrewAI investigation |
| 8 | **Multi-turn memory (turn 1)** — user introduces themselves |
| 9 | **Multi-turn memory (turn 2)** — follow-up recalling previous info |
| 10 | **Memory isolation** — proves User B doesn't see User A's data |

> **Tip for instructor demo:** The escalation cells (6 & 7) are the most impressive — they show the CrewAI crew doing a multi-agent investigation with Complaint Analyst, Account Investigator, and Resolution Specialist.

---

## Step 7 — Orchestration & Routing Tests (70 tests)

These tests verify severity-based routing, cross-department handoffs, transfer keywords, and edge cases.

```powershell
python test_orchestration.py
```

> **Note:** Requires ~5 minutes and ~70 OpenAI API calls. Shows a Rich summary table on completion.

The orchestration test categories:

| Category | Tests |
|----------|-------|
| Normal routing (all 8 departments) | 8 |
| Cross-department forced transfers | 6 |
| Escalation severity boundary (low/medium/high) | 8 |
| Transfer keyword detection (22 phrases) | 8 |
| Generic "customer service" (no history) → default dept | 3 |
| Chain transfers (A→B→C) | 3 |
| Ambiguous/tricky queries | 7 |
| `run_query_for_department()` forced | 4 |
| No "contact customer service" in responses | 5 |
| Multi-turn follow-ups | 5 |
| Escalation + edge cases | 10 |
| Product→Orders retry (3 runs) | 3 |
| **Total** | **70** |

To re-run a quick subset (8 normal routing tests):

```powershell
python -c "
from agents import run_query, compile_graph
agent = compile_graph(use_memory=False)
queries = [
    ('I want to return my order', 'orders_returns'),
    ('Why was I charged twice?', 'billing_payments'),
    ('Where is my package?', 'shipping_delivery'),
    ('Do you have this in size M?', 'product_inquiries'),
    ('Can I check my annual leave?', 'hr'),
    ('My VPN is not connecting', 'it_helpdesk'),
    ('What is the warehouse shift schedule?', 'operations'),
    ('How do I redeem loyalty points?', 'loyalty_programme'),
]
for q, expected in queries:
    r = run_query(q, user_id='test', agent=agent)
    ok = r['department'] == expected
    print(f'  [{\"PASS\" if ok else \"FAIL\"}] {expected}: {r[\"department\"]}  sev={r[\"severity\"]}')
print('Done')
"
```

**Expected:** All 8 show `[PASS]` with correct department routing.

---

## Quick Reference — All Commands in One Block

```powershell
# Activate environment
conda activate shop4you
cd ..

# Step 0: Sanity check
python -c "from config import *; from agents import compile_graph; print('OK')"

# Step 1: Data check
python -c "import json,os; total=sum(len(json.load(open(f'data/{f}')).get('entries',[])) for f in os.listdir('data') if f.endswith('.json')); print(f'QA pairs: {total}')"

# Steps 2-4: All 13 pytest tests in one command (recommended)
python -m pytest test_rag.py test_agent.py test_memory_isolation.py -v --tb=short

# Or run separately:
# python test_rag.py            # 5 RAG tests (standalone)
# python test_agent.py           # 7 E2E tests (standalone)
# python test_memory_isolation.py # memory isolation (standalone)

# Step 7: 70 orchestration & routing tests (~5 min, needs API key)
python test_orchestration.py

# 10-scenario Rich demo
python demo_customer_showcase.py

# Step 5: API server
Remove-Item shop4you_memory.db -Force -ErrorAction SilentlyContinue
python -m uvicorn app:app --host 127.0.0.1 --port 8000
# (open new terminal for API tests, then Ctrl+C to stop)

# Step 6: Notebook
# Open demo.ipynb → Run All
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Make sure you ran `conda activate shop4you` |
| `'charmap' codec can't encode` | Already fixed — if it returns, check `vector_store.py` for non-ASCII characters |
| `Collection does not exist` when starting API | Delete `shop4you_memory.db` before starting the server |
| `uvicorn: not found` | Use `python -m uvicorn` instead of bare `uvicorn` |
| CrewAI escalation tests slow (~20s) | Normal — the 3-agent crew runs sequentially via OpenAI calls |
| Notebook kernel not found | Select the `shop4you` conda env as the Jupyter kernel |
