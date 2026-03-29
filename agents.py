"""
Core LangGraph agent for Shop4You.

The graph looks like this:

  START -> classifier_agent --+--> department_rag_agent -> reflection_agent --+--> END
                              |                                              '--> retry (loop)
                              '--> escalation_agent (CrewAI investigation) -> END

What it covers:
  - Structured-output query classification (department + sentiment + severity)
  - Smart routing: only HIGH-severity negative queries escalate;
    low/medium negative goes through RAG so the department agent can help
  - External vs internal department guardrails in the classifier
  - Per-department RAG with metadata-filtered retrieval
  - Reflection loop that retries if quality is too low
  - CrewAI-powered escalation with 3-agent investigation crew
  - SQLite-backed persistent memory per user
  - Extra tool calls for Product, Loyalty & Order queries

7 Agents:
  LangGraph Orchestration Agents (this file):
     Agent 1  --  classifier_agent: Classifies queries by department, sentiment & severity
     Agent 2  --  department_rag_agent: Retrieves context from ChromaDB, calls tools, generates answers
     Agent 3  --  reflection_agent: Quality-checks answers (score 1-10), regenerates if < 7
     Agent 4  --  escalation_agent: Generates ESC reference, triggers CrewAI crew
  CrewAI Escalation Crew (escalation_crew.py)  --  only for high-severity cases:
     Agent 5  --  complaint_analyst_agent: Extracts core issue, urgency, key details
     Agent 6  --  account_investigator_agent: Builds customer context, flags VIP/repeat issues
     Agent 7  --  resolution_specialist_agent: Proposes resolution, writes final report

  Helper functions (NOT agents):
     route_query: Conditional edge  --  decides RAG path vs escalation
     should_continue_reflection: Conditional edge  --  quality gate loop control
     build_graph: Assembles the StateGraph with all nodes and edges
     compile_graph: Compiles the graph with optional memory backends
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from typing import Annotated, Literal, TypedDict

log = logging.getLogger("shop4you")

from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel, Field

from config import (
    OPENAI_API_KEY,
    LLM_MODEL,
    LLM_TEMPERATURE,
    MAX_REFLECTION_ATTEMPTS,
    ALL_DEPARTMENT_KEYS,
    DEPARTMENTS,
    SQLITE_MEMORY_DB,
    CREWAI_ESCALATION_ENABLED,
)
from prompts import (
    CLASSIFICATION_PROMPT,
    RAG_PROMPT,
    REFLECTION_PROMPT,
    REGENERATION_PROMPT,
    ESCALATION_MESSAGE_TEMPLATE,
    get_rag_prompt_vars,
    get_classification_vars,
)
from tools import (
    ALL_TOOLS,
    search_knowledge_base,
    escalate_to_human,
    search_product,
    check_loyalty_points,
    lookup_orders,
)


# ---------- State schema ----------
class Shop4YouState(TypedDict):
    """Everything we track as the query moves through the graph."""
    messages: Annotated[list, add_messages]   # conversation so far
    customer_query: str                       # raw user text
    department: str                           # which department we routed to
    sentiment: str                            # positive / neutral / negative
    severity: str                             # low / medium / high
    retrieved_context: str                    # RAG context from the KB
    reflection_count: int                     # number of reflection passes
    quality_score: int                        # latest quality score (1-10)
    final_response: str                       # what we send back to the user
    user_id: str                              # who's asking
    escalation_info: dict                     # filled in if we escalate
    classification_reasoning: str             # why the router picked this dept
    suggested_transfer: str                   # dept key if agent suggests handoff


# ---------- Structured output models ----------
class QueryClassification(BaseModel):
    """What the LLM returns when it classifies a user query."""
    department: str = Field(
        description="The department key that should handle this query. "
        f"One of: {ALL_DEPARTMENT_KEYS + ['unknown']}"
    )
    sentiment: Literal["positive", "neutral", "negative"] = Field(
        description="The emotional tone of the user's message."
    )
    severity: Literal["low", "medium", "high"] = Field(
        description=(
            "How severe/urgent the issue is. "
            "low = general inquiry or mild frustration, "
            "medium = clear dissatisfaction but manageable, "
            "high = very angry, abusive, threatening, or urgent safety/fraud issue."
        ),
        default="low",
    )
    reasoning: str = Field(
        description="Brief explanation of why this classification was chosen."
    )


class QualityCheck(BaseModel):
    """What the LLM returns when it checks the quality of a response."""
    is_relevant: bool = Field(description="Does the response address the user's query?")
    is_grounded: bool = Field(description="Is the response based on the retrieved context, not hallucinated?")
    is_complete: bool = Field(description="Is the response complete and actionable?")
    quality_score: int = Field(description="Overall quality 1-10", ge=1, le=10)
    feedback: str = Field(description="Brief feedback explaining the score")


# ---------- LLM instances ----------
llm = ChatOpenAI(
    model=LLM_MODEL,
    temperature=LLM_TEMPERATURE,
    openai_api_key=OPENAI_API_KEY,
    max_retries=5,
)

classifier_llm = llm.with_structured_output(QueryClassification)
quality_checker = llm.with_structured_output(QualityCheck)


# ---------- Agent 1: Classifies queries  --  detects department, sentiment & severity ----------
def classifier_agent(state: Shop4YouState) -> dict:
    """Figures out which department should handle the query and how the user feels.

    If the department is already set (e.g. from a cross-department handoff),
    we skip classification entirely and keep the pre-set values.
    """
    # Skip classification if this is a manual transfer (department already set)
    existing_dept = state.get("department", "")
    if existing_dept and existing_dept in ALL_DEPARTMENT_KEYS:
        log.info(
            "Skipping classification  --  department pre-set to '%s' (manual transfer)",
            existing_dept,
        )
        messages = state["messages"]
        last_message = messages[-1].content if messages else ""
        return {"customer_query": last_message}

    messages = state["messages"]
    last_message = messages[-1].content if messages else ""

    # Build conversation history from recent messages so the classifier
    # can understand follow-up queries like "pass me to customer service"
    conversation_history = ""
    if len(messages) > 1:
        recent = messages[-6:]  # last 3 turns (human+AI pairs)
        history_lines = []
        for msg in recent[:-1]:  # everything except the current message
            role = "Customer" if isinstance(msg, HumanMessage) else "Agent"
            content = msg.content[:200]  # truncate long messages
            history_lines.append(f"{role}: {content}")
        if history_lines:
            conversation_history = (
                "Recent conversation history:\n"
                + "\n".join(history_lines)
                + "\n\n"
            )

    prompt_vars = get_classification_vars(last_message, conversation_history)
    formatted = CLASSIFICATION_PROMPT.format_messages(**prompt_vars)
    result = classifier_llm.invoke(formatted)

    log.info(
        "Classified -> dept=%s  sentiment=%s  severity=%s  reason=%s",
        result.department, result.sentiment, result.severity, result.reasoning[:80],
    )

    return {
        "customer_query": last_message,
        "department": result.department,
        "sentiment": result.sentiment,
        "severity": result.severity,
        "classification_reasoning": result.reasoning,
    }


# ---------- Routing logic ----------
def route_query(state: Shop4YouState) -> str:
    """
    Decides where the query goes after classification.

    Routing rules (in priority order):
      1. Unknown/invalid department            -> escalate
      2. Negative sentiment + HIGH severity    -> escalate (very angry / abusive)
      3. Negative sentiment + low/medium sev.  -> RAG  (let the dept agent handle it)
      4. Everything else                       -> RAG

    This prevents mild complaints ("a bit disappointed") from skipping
    the knowledge base entirely.
    """
    dept = state.get("department", "unknown")
    sentiment = state.get("sentiment", "neutral")
    severity = state.get("severity", "low")

    # Unknown department -> escalate
    if dept == "unknown" or dept not in ALL_DEPARTMENT_KEYS:
        log.info("Routing -> escalate (unknown department)")
        return "escalate"

    # Only escalate negative sentiment when severity is HIGH
    if sentiment == "negative" and severity == "high":
        log.info("Routing -> escalate (negative + high severity)")
        return "escalate"

    # Everything else (including mild/medium negative) goes through RAG
    log.info("Routing -> department_rag (%s, sentiment=%s, severity=%s)", dept, sentiment, severity)
    return "department_rag"


# ---------- Agent 2: Retrieves context from ChromaDB, calls tools, generates answers ----------
def department_rag_agent(state: Shop4YouState) -> dict:
    """
    Grabs context from the knowledge base and generates an answer.

    For the Product and Loyalty departments we also call specialised tools
    (product search / loyalty lookup) when the query looks like it warrants one.
    """
    query = state["customer_query"]
    department = state["department"]

    # Detect "connect me / transfer me / chat with" requests
    _transfer_keywords = [
        "connect me", "transfer me", "speak to", "talk to", "chat with",
        "put me through", "redirect me", "route me", "switch me",
        "can i speak", "can i talk", "can i chat", "let me talk",
        "i need to speak", "i want to speak", "i want to talk",
        "connect to", "transfer to", "pass me to", "pass me",
        "customer service", "speak with", "talk with",
    ]
    query_lower = query.lower()
    is_transfer = any(kw in query_lower for kw in _transfer_keywords)

    # Start with the knowledge base lookup
    context = search_knowledge_base.invoke({"query": query, "department_key": department})
    log.debug("KB context length: %d chars", len(context))

    # If the department has an extra tool, try calling it too
    extra_context = ""
    if department == "product_inquiries":
        extra_context = _try_product_search(query)
    elif department == "loyalty_programme":
        extra_context = _try_loyalty_lookup(query)
    elif department == "orders_returns":
        extra_context = _try_order_lookup(query, state.get("user_id", ""))

    # Merge any extra tool output into the context
    if extra_context:
        context = f"{context}\n\n--- Tool Results ---\n{extra_context}"

    # For transfer/connect requests, inject a department intro so the agent
    # welcomes the user instead of giving the "not found" fallback
    if is_transfer:
        dept_info = DEPARTMENTS.get(department, {})
        dept_name = dept_info.get("name", department)
        examples = dept_info.get("example_topics", [])
        example_str = ", ".join(examples[:4]) if examples else "general queries"
        transfer_intro = (
            f"--- Department Introduction ---\n"
            f"You are the {dept_name} Agent. The user has asked to be connected "
            f"to your department. Greet them, introduce yourself, and ask what "
            f"specific issue they need help with. Example topics you handle: "
            f"{example_str}.\n"
            f"Do NOT redirect them  --  YOU are the agent they asked to speak to."
        )
        context = f"{transfer_intro}\n\n{context}"
        log.info("Transfer request detected -> injecting %s intro", dept_name)

    # Build the final answer with the RAG template
    prompt_vars = get_rag_prompt_vars(department, query, context)
    formatted = RAG_PROMPT.format_messages(**prompt_vars)
    response = llm.invoke(formatted)

    # Check if the agent is suggesting a cross-department handoff
    import re as _re
    response_text = response.content
    suggested_transfer = ""
    transfer_match = _re.search(r"\[SUGGEST_TRANSFER:\s*(\w+)\]", response_text)
    if transfer_match:
        suggested_dept = transfer_match.group(1).strip()
        if suggested_dept in ALL_DEPARTMENT_KEYS and suggested_dept != department:
            suggested_transfer = suggested_dept
            # Remove the machine-readable tag from the user-facing response
            response_text = _re.sub(r"\n?\[SUGGEST_TRANSFER:\s*\w+\]\n?", "", response_text).strip()
            log.info(
                "Cross-dept handoff suggested: %s -> %s",
                department, suggested_transfer,
            )

    # Fallback: if no tag but the response mentions a different department
    # by name in a "transfer" / "reach out" / "contact" context, detect it.
    if not suggested_transfer:
        resp_lower = response_text.lower()
        _transfer_hints = [
            "reach out to", "contact the", "transfer you to",
            "better handled by", "falls under", "recommend reaching out",
            "suggest contacting", "handled by our", "direct you to",
        ]
        if any(hint in resp_lower for hint in _transfer_hints):
            # Find the department that is most prominently mentioned
            best_dept = ""
            best_score = 0
            for dept_key, dept_info in DEPARTMENTS.items():
                if dept_key == department:
                    continue
                dept_name_lower = dept_info["name"].lower()
                key_as_words = dept_key.replace("_", " ")
                score = resp_lower.count(dept_name_lower) + resp_lower.count(key_as_words)
                if score > best_score:
                    best_score = score
                    best_dept = dept_key
            if best_dept:
                suggested_transfer = best_dept
                log.info(
                    "Cross-dept handoff detected (fallback): %s -> %s",
                    department, suggested_transfer,
                )

    return {
        "retrieved_context": context,
        "final_response": response_text,
        "messages": [AIMessage(content=response_text)],
        "reflection_count": 0,
        "quality_score": 0,
        "suggested_transfer": suggested_transfer,
    }


def _try_product_search(query: str) -> str:
    """Fires the product-search tool if the query looks product-related."""
    product_keywords = [
        "jumper", "shoes", "wallet", "headphones", "t-shirt", "shirt",
        "jacket", "trainers", "dress", "bag", "watch", "stock", "available",
        "do you have", "in stock", "buy", "purchase",
    ]
    query_lower = query.lower()
    if any(kw in query_lower for kw in product_keywords):
        return search_product.invoke({"product_name": query})
    return ""


def _try_loyalty_lookup(query: str) -> str:
    """Looks for an employee ID pattern (EMP###) and runs the loyalty tool."""
    import re
    match = re.search(r"EMP\d{3}", query, re.IGNORECASE)
    if match:
        return check_loyalty_points.invoke({"employee_id": match.group()})
    return ""


def _try_order_lookup(query: str, user_id: str) -> str:
    """Looks up orders for the current user when the query is about orders."""
    if not user_id or "@" not in user_id:
        return ""
    order_keywords = [
        "order", "delivery", "return", "refund", "tracking", "shipped",
        "delivered", "purchase", "bought", "status", "parcel", "package",
        "when did", "last order", "my order", "recent order",
    ]
    query_lower = query.lower()
    if any(kw in query_lower for kw in order_keywords):
        return lookup_orders.invoke({"customer_email": user_id})
    # Even without keywords, always provide order summary for orders_returns dept
    return lookup_orders.invoke({"customer_email": user_id})


# ---------- Agent 3: Quality-checks answers (score 1-10), regenerates if < 7 ----------
def reflection_agent(state: Shop4YouState) -> dict:
    """
    Scores the generated answer for quality.
    If it's below 7 and we haven't hit the retry limit, we regenerate.
    """
    response = state.get("final_response", "")
    query = state.get("customer_query", "")
    context = state.get("retrieved_context", "")
    department = state.get("department", "")
    count = state.get("reflection_count", 0)

    # Run the quality checker
    reflection_vars = {
        "query": query,
        "context": context[:1500],
        "response": response,
    }
    formatted = REFLECTION_PROMPT.format_messages(**reflection_vars)
    check = quality_checker.invoke(formatted)

    # Score is fine (or we've used up our retries)  --  accept as-is
    if check.quality_score >= 7 or count >= MAX_REFLECTION_ATTEMPTS:
        log.info(
            "Reflection pass %d -> score=%d  (accepted)",
            count + 1, check.quality_score,
        )
        return {
            "reflection_count": count + 1,
            "quality_score": check.quality_score,
        }

    # Quality too low  --  regenerate with the feedback
    log.info(
        "Reflection pass %d -> score=%d  (regenerating: %s)",
        count + 1, check.quality_score, check.feedback[:60],
    )
    dept_name = DEPARTMENTS.get(department, {}).get("name", department)
    regen_vars = {
        "department_name": dept_name,
        "feedback": check.feedback,
        "query": query,
        "context": context,
    }
    regen_formatted = REGENERATION_PROMPT.format_messages(**regen_vars)
    improved = llm.invoke(regen_formatted)

    return {
        "final_response": improved.content,
        "messages": [AIMessage(content=improved.content)],
        "reflection_count": count + 1,
        "quality_score": check.quality_score,
    }


def should_continue_reflection(state: Shop4YouState) -> str:
    """
    Decides whether to loop back for another reflection pass or wrap up.
    Stops when quality >= 7 or we've exhausted our retry budget.
    """
    score = state.get("quality_score", 10)
    count = state.get("reflection_count", 0)
    if score >= 7 or count >= MAX_REFLECTION_ATTEMPTS:
        return "done"
    return "retry"


# ---------- Agent 4: Generates ESC reference, triggers CrewAI crew for high-severity cases ----------
def escalation_agent(state: Shop4YouState) -> dict:
    """
    Hands the conversation off to a human agent.

    When CREWAI_ESCALATION_ENABLED is True, a 3-agent CrewAI crew runs
    first to investigate the complaint and produce a structured report.
    The report is appended to the escalation info so the human agent
    has all the context they need.
    """
    query = state.get("customer_query", "")
    sentiment = state.get("sentiment", "unknown")
    department = state.get("department", "unknown")
    dept_name = DEPARTMENTS.get(department, {}).get(
        "name", department.replace("_", " ").title()
    )

    reason = f"Sentiment: {sentiment}, Severity: {state.get('severity', 'unknown')}, Department: {department}"
    ref_number = f"ESC-{abs(hash(query)) % 100000:05d}"
    timestamp = datetime.now(timezone.utc).isoformat()

    log.warning(
        "Escalating query -> ref=%s  sentiment=%s  dept=%s",
        ref_number, sentiment, department,
    )

    # Run the CrewAI investigation crew if enabled
    crew_report = {}
    if CREWAI_ESCALATION_ENABLED:
        try:
            from escalation_crew import run_escalation_crew

            log.info("Starting CrewAI escalation investigation...")
            crew_report = run_escalation_crew(
                query=query,
                sentiment=sentiment,
                department=department,
                reference_number=ref_number,
            )
            log.info("CrewAI investigation complete for ref=%s", ref_number)
        except Exception as e:
            log.error("CrewAI escalation failed: %s  --  falling back to basic escalation", e)
            crew_report = {"error": str(e)}

    # Build the user-facing message
    escalation_msg = ESCALATION_MESSAGE_TEMPLATE.format(
        department_name=dept_name,
        reference_number=ref_number,
    )

    # If the crew produced a full report, include the resolution summary
    # in the user-facing response so they know what to expect
    crew_output = crew_report.get("crew_output", "")
    if crew_output and "error" not in crew_report:
        escalation_msg += (
            "\n\n---\n"
            "**Escalation Investigation Report**\n\n"
            f"{crew_output}"
        )

    return {
        "final_response": escalation_msg,
        "messages": [AIMessage(content=escalation_msg)],
        "escalation_info": {
            "reason": reason,
            "query": query,
            "sentiment": sentiment,
            "department": department,
            "department_name": dept_name,
            "reference_number": ref_number,
            "timestamp": timestamp,
            "status": "pending",
            "crew_report": crew_report,
        },
    }


# ---------- Wire up the full graph ----------
def build_graph() -> StateGraph:
    """
    Puts together the Shop4You workflow graph.

    START
      -> classifier_agent
      -> [conditional]
          |-- department_rag_agent -> reflection_agent -> [quality gate]
          |                                                |-- done -> END
          |                                                '-- retry -> reflection_agent (loop)
          '-- escalation_agent -> END
    """
    graph = StateGraph(Shop4YouState)

    # Agent 1: classifier_agent  --  detects department, sentiment & severity
    graph.add_node("classify_query", classifier_agent)
    # Agent 2: department_rag_agent  --  retrieves context, calls tools, generates answers
    graph.add_node("department_rag", department_rag_agent)
    # Agent 3: reflection_agent  --  quality-checks answers, regenerates if < 7
    graph.add_node("reflect", reflection_agent)
    # Agent 4: escalation_agent  --  generates ESC ref, triggers CrewAI crew
    graph.add_node("escalate", escalation_agent)

    # Entry
    graph.add_edge(START, "classify_query")

    # After classification, either RAG or escalation
    graph.add_conditional_edges(
        "classify_query",
        route_query,
        {
            "department_rag": "department_rag",
            "escalate": "escalate",
        },
    )

    # RAG node feeds into the reflection step
    graph.add_edge("department_rag", "reflect")

    # Reflection can either finish or loop for another attempt
    graph.add_conditional_edges(
        "reflect",
        should_continue_reflection,
        {
            "done": END,
            "retry": "reflect",     # loop back
        },
    )

    # Escalation is a dead end (goes straight to END)
    graph.add_edge("escalate", END)

    return graph


# ---------- Compile the graph (NOT an agent  --  just a graph compiler) ----------
def compile_graph(use_memory: bool = True, persist: bool = False):
    """
    Compiles the LangGraph workflow into a runnable graph.

    This is a factory function, NOT an agent. It assembles the graph
    built by build_graph() and attaches an optional memory backend.

    use_memory: turns on conversation checkpointing.
    persist:    True  -> SQLite file on disk (survives restarts)
                False -> in-memory (faster, good for tests)
    """
    graph = build_graph()

    if not use_memory:
        return graph.compile()

    if persist:
        # Disk-backed memory so conversations survive restarts
        conn = sqlite3.connect(SQLITE_MEMORY_DB, check_same_thread=False)
        memory = SqliteSaver(conn)
        return graph.compile(checkpointer=memory)

    # Default: in-memory (quicker for dev / testing)
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


# ---------- Quick way to run a single query ----------
def run_query(query: str, user_id: str = "default_user", agent=None, user_context: str = ""):
    """Sends one query through the agent and returns a tidy results dict.

    user_context: optional string like '[Customer: Sarah | Email: sarah@gmail.com | Tier: Gold]'
                  that is prepended to the query so the LLM knows who is asking.
    """
    if agent is None:
        agent = compile_graph(use_memory=True)

    config = {"configurable": {"thread_id": f"user_{user_id}"}}

    # Prepend user context so the LLM can personalise its response
    enriched_query = f"{user_context}\n{query}" if user_context else query

    result = agent.invoke(
        {
            "messages": [HumanMessage(content=enriched_query)],
            "customer_query": enriched_query,
            "department": "",
            "sentiment": "",
            "severity": "",
            "retrieved_context": "",
            "reflection_count": 0,
            "quality_score": 0,
            "final_response": "",
            "user_id": user_id,
            "escalation_info": {},
            "classification_reasoning": "",
            "suggested_transfer": "",
        },
        config=config,
    )

    return {
        "response": result.get("final_response", ""),
        "department": result.get("department", ""),
        "sentiment": result.get("sentiment", ""),
        "severity": result.get("severity", ""),
        "quality_score": result.get("quality_score", 0),
        "classification_reasoning": result.get("classification_reasoning", ""),
        "escalation_info": result.get("escalation_info", {}),
        "suggested_transfer": result.get("suggested_transfer", ""),
    }


# ---------- Run a query forced to a specific department ----------
def run_query_for_department(
    query: str,
    target_department: str,
    user_id: str = "default_user",
    agent=None,
    user_context: str = "",
) -> dict:
    """Re-runs a query through a specific department, skipping classification.

    Used when the original department agent suggests a cross-department
    handoff and the user accepts.
    """
    if agent is None:
        agent = compile_graph(use_memory=True)

    config = {"configurable": {"thread_id": f"user_{user_id}"}}
    enriched_query = f"{user_context}\n{query}" if user_context else query

    result = agent.invoke(
        {
            "messages": [HumanMessage(content=enriched_query)],
            "customer_query": enriched_query,
            "department": target_department,   # <-- pre-set the department
            "sentiment": "neutral",            # reset for the new agent
            "severity": "low",
            "retrieved_context": "",
            "reflection_count": 0,
            "quality_score": 0,
            "final_response": "",
            "user_id": user_id,
            "escalation_info": {},
            "classification_reasoning": f"Manual transfer to {target_department}",
            "suggested_transfer": "",
        },
        config=config,
    )

    return {
        "response": result.get("final_response", ""),
        "department": result.get("department", ""),
        "sentiment": result.get("sentiment", ""),
        "severity": result.get("severity", ""),
        "quality_score": result.get("quality_score", 0),
        "classification_reasoning": result.get("classification_reasoning", ""),
        "escalation_info": result.get("escalation_info", {}),
        "suggested_transfer": result.get("suggested_transfer", ""),
    }
