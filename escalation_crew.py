"""
CrewAI-powered escalation investigation crew.

When a query gets escalated (negative sentiment or unknown department),
this crew runs a multi-agent investigation before handing off to a
human agent. Three specialised agents collaborate:

  Agent 5  --  complaint_analyst_agent:       Summarises the issue, extracts key details
  Agent 6  --  account_investigator_agent:    Looks up customer context, order history, past tickets
  Agent 7  --  resolution_specialist_agent:   Proposes a concrete resolution (refund, replacement, callback)

The final output is a structured escalation report that gives the human
agent everything they need to pick up the case immediately.
"""

import logging
import os
from datetime import datetime, timezone

from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv

from config import DEPARTMENTS, OPENAI_API_KEY

load_dotenv()

log = logging.getLogger("shop4you.crew")


# -- Shared LLM config ----------------------------------------
# CrewAI picks up the model from the agent definition.
# We use gpt-4o to keep it consistent with the LangGraph pipeline.
CREW_LLM = "gpt-4o"


# -- Agent definitions ----------------------------------------

# Agent 5: Extracts core issue, urgency, emotional state, key details
def _build_complaint_analyst_agent() -> Agent:
    """Agent 5: tears the complaint apart and pulls out structured details."""
    return Agent(
        role="Complaint Analyst",
        goal=(
            "Analyse the customer or employee complaint thoroughly. "
            "Extract the core issue, urgency level, affected product or service, "
            "and any specific details (order numbers, dates, names) mentioned."
        ),
        backstory=(
            "You are a senior complaint analyst at Shop4You with 8 years of "
            "experience in retail customer service. You've seen every type of "
            "complaint  --  from missing parcels to billing disputes to workplace "
            "grievances. Your job is to cut through the noise and extract the "
            "facts that matter so the next person in the chain can act fast."
        ),
        llm=CREW_LLM,
        verbose=False,
        allow_delegation=False,
    )


# Agent 6: Builds customer context, identifies responsible dept, flags VIP/repeat issues
def _build_account_investigator_agent() -> Agent:
    """Agent 6: builds a picture of the customer's context and history."""
    return Agent(
        role="Account Investigator",
        goal=(
            "Based on the complaint details, build a profile of the user's "
            "likely account context  --  identify what department would normally "
            "handle this, whether there might be open orders or tickets, "
            "and flag any VIP or priority indicators."
        ),
        backstory=(
            "You work in Shop4You's escalation support team. When a complaint "
            "comes in that the AI couldn't resolve, you dig into the context. "
            "You know the department structure inside out and can quickly figure "
            "out where the ball was dropped. You also know that repeat "
            "complainers and VIP customers need special handling."
        ),
        llm=CREW_LLM,
        verbose=False,
        allow_delegation=False,
    )


# Agent 7: Proposes resolution + goodwill gesture, writes final escalation report
def _build_resolution_specialist_agent() -> Agent:
    """Agent 7: proposes a concrete resolution and writes the final report."""
    return Agent(
        role="Resolution Specialist",
        goal=(
            "Recommend a specific resolution for this escalated case. "
            "Provide clear next steps for the human agent: what action to "
            "take, what to offer the customer, and what follow-up is needed. "
            "Compile everything into a clean escalation report."
        ),
        backstory=(
            "You are Shop4You's head of customer resolution. Your decisions "
            "carry weight  --  you can authorise refunds, arrange replacements, "
            "schedule priority callbacks, and approve goodwill gestures. "
            "You always balance customer satisfaction with business sense. "
            "Your reports are the ones human agents actually read, so they "
            "need to be concise and actionable."
        ),
        llm=CREW_LLM,
        verbose=False,
        allow_delegation=False,
    )


# -- Task definitions -----------------------------------------

def _build_analysis_task(agent: Agent, query: str, sentiment: str, department: str) -> Task:
    """Task 1: Analyse the complaint."""
    dept_info = DEPARTMENTS.get(department, {})
    dept_name = dept_info.get("name", department.replace("_", " ").title())
    audience = dept_info.get("audience", "unknown")

    return Task(
        description=(
            f"A user query has been escalated.\n\n"
            f"User query: \"{query}\"\n"
            f"Detected sentiment: {sentiment}\n"
            f"Detected department: {dept_name} ({audience})\n\n"
            f"Your job:\n"
            f"1. Identify the CORE ISSUE in 1-2 sentences.\n"
            f"2. Extract any specific details mentioned (order IDs, dates, product names, "
            f"employee IDs, amounts).\n"
            f"3. Rate the URGENCY on a scale: low / medium / high / critical.\n"
            f"4. Identify the EMOTIONAL STATE of the user.\n"
            f"5. List any unanswered questions the user seems to have."
        ),
        expected_output=(
            "A structured complaint analysis with: core issue, extracted details, "
            "urgency level, emotional state, and unanswered questions."
        ),
        agent=agent,
    )


def _build_investigation_task(agent: Agent, query: str, department: str) -> Task:
    """Task 2: Investigate the account context."""
    dept_info = DEPARTMENTS.get(department, {})
    dept_name = dept_info.get("name", department.replace("_", " ").title())
    audience = dept_info.get("audience", "unknown")

    return Task(
        description=(
            f"Based on the complaint analysis from the previous step, investigate "
            f"the account context.\n\n"
            f"Original query: \"{query}\"\n"
            f"Department: {dept_name}\n"
            f"User type: {'External customer' if audience == 'external' else 'Internal employee'}\n\n"
            f"Your job:\n"
            f"1. Determine which Shop4You department(s) should handle this case.\n"
            f"2. Identify if this is an {'order/billing/shipping issue' if audience == 'external' else 'HR/IT/facilities issue'}.\n"
            f"3. Flag if the user might be a VIP, repeat customer, or long-term employee "
            f"based on context clues.\n"
            f"4. Note any red flags (potential fraud, policy violation, safety issue).\n"
            f"5. Suggest what information the human agent should look up first."
        ),
        expected_output=(
            "Account investigation summary with: responsible department(s), issue "
            "category, priority flags, red flags, and recommended first-look items "
            "for the human agent."
        ),
        agent=agent,
    )


def _build_resolution_task(agent: Agent, reference_number: str) -> Task:
    """Task 3: Propose resolution and write the final report."""
    return Task(
        description=(
            f"Based on the complaint analysis and account investigation, "
            f"write the final escalation report.\n\n"
            f"Reference number: {reference_number}\n\n"
            f"Your job:\n"
            f"1. Recommend a PRIMARY RESOLUTION (refund, replacement, callback, "
            f"   apology, policy clarification, IT fix, etc.).\n"
            f"2. Suggest a GOODWILL GESTURE if appropriate (discount code, loyalty "
            f"   points, priority shipping on next order, etc.).\n"
            f"3. Outline FOLLOW-UP STEPS for the human agent.\n"
            f"4. Set an expected RESPONSE TIME (urgent: 1 hour, high: 4 hours, "
            f"   medium: 24 hours, low: 48 hours).\n"
            f"5. Compile everything into a clean, ready-to-use escalation report."
        ),
        expected_output=(
            f"A complete escalation report for case {reference_number} with sections: "
            f"Summary, Recommended Resolution, Goodwill Gesture, Follow-up Steps, "
            f"Response Time, and Agent Notes."
        ),
        agent=agent,
    )


# -- Main entry point -----------------------------------------

def run_escalation_crew(
    query: str,
    sentiment: str,
    department: str,
    reference_number: str,
) -> dict:
    """
    Spins up the 3-agent investigation crew for an escalated query.

    Returns a dict with the crew's findings:
      - analysis:   complaint breakdown from Agent 1
      - investigation: account context from Agent 2
      - resolution: proposed fix + final report from Agent 3
      - crew_output: the final consolidated output
    """
    log.info(
        "CrewAI escalation started -> ref=%s dept=%s sentiment=%s",
        reference_number, department, sentiment,
    )

    # Build agents
    complaint_analyst_agent = _build_complaint_analyst_agent()
    account_investigator_agent = _build_account_investigator_agent()
    resolution_specialist_agent = _build_resolution_specialist_agent()

    # Build tasks (sequential  --  each builds on the previous)
    analysis_task = _build_analysis_task(complaint_analyst_agent, query, sentiment, department)
    investigation_task = _build_investigation_task(account_investigator_agent, query, department)
    resolution_task = _build_resolution_task(resolution_specialist_agent, reference_number)

    # Assemble the crew
    crew = Crew(
        agents=[complaint_analyst_agent, account_investigator_agent, resolution_specialist_agent],
        tasks=[analysis_task, investigation_task, resolution_task],
        process=Process.sequential,
        verbose=False,
    )

    # Run it
    result = crew.kickoff()

    log.info("CrewAI escalation completed -> ref=%s", reference_number)

    # Pull individual task outputs if available
    task_outputs = {}
    try:
        if hasattr(result, "tasks_output") and result.tasks_output:
            for i, label in enumerate(["analysis", "investigation", "resolution"]):
                if i < len(result.tasks_output):
                    task_outputs[label] = str(result.tasks_output[i])
    except Exception:
        # Not critical  --  we still have the final output
        pass

    return {
        "analysis": task_outputs.get("analysis", ""),
        "investigation": task_outputs.get("investigation", ""),
        "resolution": task_outputs.get("resolution", ""),
        "crew_output": str(result),
        "reference_number": reference_number,
    }
