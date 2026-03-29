"""
FastAPI service that wraps the Shop4You agent.

Endpoints:
  POST /chat           -- send a query, get a response
  POST /transfer       -- re-run a query through a specific department (handoff)
  GET  /history/{uid}  -- conversation history for a user
  DELETE /history/{uid} -- wipe a user's history
  GET  /departments    -- list all departments
  GET  /health         -- quick health check

Start it:  uvicorn app:app --reload
Swagger:   http://localhost:8000/docs
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from agents import compile_graph, run_query, run_query_for_department
from config import DEPARTMENTS


# Compile the agent once at startup so we don't redo it per request.
_agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent
    _agent = compile_graph(use_memory=True, persist=True)
    yield


# ---------- App setup ----------
app = FastAPI(
    title="Shop4You AI Assistant",
    description="Agentic AI Customer & Employee Support for Shop4You",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------- Request / response schemas ------------
class ChatRequest(BaseModel):
    query: str = Field(..., description="The user's question or message", min_length=1)
    user_id: str = Field(default="guest", description="Unique user identifier for session tracking")
    user_context: str = Field(default="", description="Optional user context (e.g. name, tier) for personalised responses")


class ChatResponse(BaseModel):
    response: str
    department: str
    sentiment: str
    severity: str = "low"
    quality_score: int = 0
    escalated: bool
    reference_number: Optional[str] = None
    classification_reasoning: Optional[str] = None
    suggested_transfer: Optional[str] = None


class TransferRequest(BaseModel):
    query: str = Field(..., description="The query to re-run in the target department", min_length=1)
    target_department: str = Field(..., description="Department key to transfer to")
    user_id: str = Field(default="guest", description="User identifier")
    user_context: str = Field(default="", description="Optional user context")


class DepartmentInfo(BaseModel):
    key: str
    name: str
    audience: str
    description: str


# ---------- Endpoints ----------

@app.get("/health")
def health_check():
    """Quick liveness check."""
    return {"status": "healthy", "service": "Shop4You AI Assistant", "version": "1.0.0"}


@app.get("/departments", response_model=list[DepartmentInfo])
def list_all_departments():
    """Returns every department with its key, audience, and description."""
    return [
        DepartmentInfo(
            key=k,
            name=v["name"],
            audience=v["audience"],
            description=v["description"],
        )
        for k, v in DEPARTMENTS.items()
    ]


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Main entry point  --  sends the user's message through the agent."""
    try:
        result = run_query(request.query, user_id=request.user_id, agent=_agent, user_context=request.user_context)

        escalation = result.get("escalation_info", {})
        is_escalated = bool(escalation)

        return ChatResponse(
            response=result.get("response", "Sorry, I could not process your query."),
            department=result.get("department", "unknown"),
            sentiment=result.get("sentiment", "unknown"),
            severity=result.get("severity", "low"),
            quality_score=result.get("quality_score", 0),
            escalated=is_escalated,
            reference_number=escalation.get("reference_number") if is_escalated else None,
            classification_reasoning=result.get("classification_reasoning"),
            suggested_transfer=result.get("suggested_transfer") or None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


@app.post("/transfer", response_model=ChatResponse)
def transfer(request: TransferRequest):
    """Re-run a query through a specific department (cross-department handoff)."""
    try:
        result = run_query_for_department(
            request.query,
            request.target_department,
            user_id=request.user_id,
            agent=_agent,
            user_context=request.user_context,
        )
        escalation = result.get("escalation_info", {})
        is_escalated = bool(escalation)
        return ChatResponse(
            response=result.get("response", "Sorry, I could not process your query."),
            department=result.get("department", "unknown"),
            sentiment=result.get("sentiment", "unknown"),
            severity=result.get("severity", "low"),
            quality_score=result.get("quality_score", 0),
            escalated=is_escalated,
            reference_number=escalation.get("reference_number") if is_escalated else None,
            classification_reasoning=result.get("classification_reasoning"),
            suggested_transfer=result.get("suggested_transfer") or None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transfer error: {str(e)}")


@app.get("/history/{user_id}")
def get_history(user_id: str):
    """Pulls the stored conversation for a given user."""
    try:
        config = {"configurable": {"thread_id": f"user_{user_id}"}}
        state = _agent.get_state(config)

        if not state or not state.values:
            return {"user_id": user_id, "messages": [], "message_count": 0}

        messages = state.values.get("messages", [])
        formatted = []
        for msg in messages:
            formatted.append({
                "role": msg.__class__.__name__.replace("Message", "").lower(),
                "content": msg.content,
            })

        return {
            "user_id": user_id,
            "messages": formatted,
            "message_count": len(formatted),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving history: {str(e)}")


@app.delete("/history/{user_id}")
def clear_history(user_id: str):
    """Resets a user's conversation so the next query starts fresh."""
    return {
        "user_id": user_id,
        "status": "cleared",
        "message": "Next query will start a fresh conversation.",
    }
