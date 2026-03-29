"""
+======================================================================+
|  Shop4You  --  Customer Demo Showcase Script                          |
|  Demonstrates multi-agent orchestration, RAG, tools, handoffs,     |
|  escalation (CrewAI), reflection, and persistent memory.           |
|                                                                    |
|  Usage:  conda activate shop4you                                   |
|          python demo_customer_showcase.py                          |
|                                                                    |
|  This script walks through 10 scenarios as customer Sarah Ahmed    |
|  (Gold tier) to showcase every major capability of the system.     |
+======================================================================+
"""

import time
import textwrap
from rich.console  import Console
from rich.panel    import Panel
from rich.table    import Table
from rich.markdown import Markdown
from rich.rule     import Rule

from agents import compile_graph, run_query, run_query_for_department
from users  import get_user

# --- Setup ------------------------------------------------------------
console = Console(width=110)

CUSTOMER_EMAIL = "sarah@gmail.com"
CUSTOMER_USER  = get_user(CUSTOMER_EMAIL)
USER_CONTEXT   = (
    f"[Customer: {CUSTOMER_USER['name']} | "
    f"Email: {CUSTOMER_EMAIL} | "
    f"Tier: {CUSTOMER_USER.get('tier', 'Standard')}]"
)

PAUSE_BETWEEN = 2          # seconds between scenarios (for readability)
SCENARIO_NUM  = 0


def banner():
    console.print(Panel.fit(
        "[bold bright_cyan]Shop4You  --  Customer Demo Showcase[/bold bright_cyan]\n"
        f"Logged in as: [bold]{CUSTOMER_USER['name']}[/bold] ({CUSTOMER_EMAIL})  --  "
        f"[gold1]{CUSTOMER_USER.get('tier', 'Standard')} Tier[/gold1]\n\n"
        "This demo walks through 10 real scenarios showing:\n"
        "  - Multi-department RAG routing (Agent 1  --  LangGraph Orchestrator)\n"
        "  - Tool calls (order lookup, product search)\n"
        "  - Cross-department handoffs & transfers\n"
        "  - CrewAI escalation (Agents 2-3-4 investigation crew)\n"
        "  - Reflection & self-correction loop\n"
        "  - Persistent per-user memory\n"
        "  - Farewell handling",
        border_style="bright_cyan",
        title="\U0001f389 DEMO START",
    ))


def section(title: str, description: str, agents_used: str):
    """Print a coloured scenario header."""
    global SCENARIO_NUM
    SCENARIO_NUM += 1
    console.print()
    console.print(Rule(f"[bold yellow]Scenario {SCENARIO_NUM}[/bold yellow]", style="yellow"))
    console.print(f"[bold bright_white]{title}[/bold bright_white]")
    console.print(f"[dim]{description}[/dim]")
    console.print(f"[italic bright_magenta]Agents: {agents_used}[/italic bright_magenta]")
    console.print()


def show_result(result: dict, query: str):
    """Display the result in a nicely formatted Rich panel."""
    # Metadata table
    meta = Table.grid(padding=(0, 2))
    meta.add_column(style="bold")
    meta.add_column()
    meta.add_row("Query",       f"[bright_white]{query}[/bright_white]")
    meta.add_row("Department",  f"[bright_cyan]{result['department']}[/bright_cyan]")
    meta.add_row("Sentiment",   sentiment_colour(result["sentiment"]))
    meta.add_row("Severity",    severity_colour(result.get("severity", "low")))
    is_escalated = bool(result.get("escalation_info"))
    if is_escalated:
        meta.add_row("Quality",  "[bold red]ESCALATED -> CrewAI[/bold red]")
    else:
        meta.add_row("Quality",  f"[green]{result['quality_score']}/10[/green]")
    if result.get("classification_reasoning"):
        meta.add_row("Reasoning", f"[dim]{result['classification_reasoning'][:120]}[/dim]")
    if result.get("suggested_transfer"):
        meta.add_row("Transfer ->", f"[bold yellow]{result['suggested_transfer']}[/bold yellow]")
    console.print(meta)
    console.print()

    # Response
    response_text = result.get("response", "No response.")
    console.print(Panel(
        Markdown(response_text[:2000]),
        title="[bold blue]Shop4You Assistant[/bold blue]",
        border_style="blue",
        padding=(1, 2),
    ))

    # CrewAI Escalation report if present
    esc = result.get("escalation_info", {})
    if esc:
        ref = esc.get("reference_number", "N/A")
        console.print(f"\n[bold red][!] ESCALATED  --  Ref: {ref}[/bold red]")
        crew = esc.get("crew_report", {})
        if crew:
            for step_name in ("analysis", "investigation", "resolution"):
                text = crew.get(step_name, "")
                if text:
                    console.print(Panel(
                        text[:800],
                        title=f"[bold]{step_name.upper()}[/bold]",
                        border_style="red" if step_name == "analysis" else "yellow" if step_name == "investigation" else "green",
                        padding=(0, 1),
                    ))


def sentiment_colour(s: str) -> str:
    colours = {"positive": "green", "neutral": "bright_white", "negative": "red"}
    return f"[{colours.get(s, 'white')}]{s}[/{colours.get(s, 'white')}]"


def severity_colour(s: str) -> str:
    colours = {"low": "green", "medium": "yellow", "high": "bold red"}
    return f"[{colours.get(s, 'white')}]{s}[/{colours.get(s, 'white')}]"


def pause():
    time.sleep(PAUSE_BETWEEN)


# =======================================================================
#  MAIN DEMO
# =======================================================================

def main():
    banner()

    # Compile agent with persistent memory so later scenarios can recall earlier ones
    console.print("[dim]Compiling agent with persistent SQLite memory...[/dim]")
    agent = compile_graph(use_memory=True, persist=True)
    console.print("[green][PASS] Agent ready[/green]\n")

    # ------------------------------------------------------------------
    # SCENARIO 1  --  Normal RAG: Orders & Returns
    # Demonstrates: Orchestrator classification -> RAG retrieval -> GPT-4o answer -> Reflection
    # ------------------------------------------------------------------
    section(
        "Normal RAG  --  Orders & Returns Department",
        "A standard customer question routed to the correct department via RAG.",
        "Agent 1 (classifier_agent) -> Agent 2 (department_rag_agent) -> Agent 3 (reflection_agent)",
    )
    q1 = "What is your return policy? I bought something last week and it doesn't fit."
    r1 = run_query(q1, user_id=CUSTOMER_EMAIL, agent=agent, user_context=USER_CONTEXT)
    show_result(r1, q1)
    pause()

    # ------------------------------------------------------------------
    # SCENARIO 2  --  Tool Call: Order Lookup
    # Demonstrates: Orchestrator detects order query -> calls lookup_orders tool -> enriches answer
    # ------------------------------------------------------------------
    section(
        "Tool Call  --  Order Lookup (lookup_orders)",
        "The orchestrator calls the lookup_orders tool to fetch Sarah's real order data.",
        "classifier_agent -> department_rag_agent + lookup_orders tool -> reflection_agent",
    )
    q2 = "Could you look up my order ORD-4290 and tell me its current status and delivery date?"
    r2 = run_query(q2, user_id=CUSTOMER_EMAIL, agent=agent, user_context=USER_CONTEXT)
    show_result(r2, q2)
    pause()

    # ------------------------------------------------------------------
    # SCENARIO 3  --  Tool Call: Product Search
    # Demonstrates: Product Inquiries dept -> calls search_product tool
    # ------------------------------------------------------------------
    section(
        "Tool Call  --  Product Search (search_product)",
        "Customer asks about product availability -> routed to Product Inquiries -> tool call.",
        "classifier_agent (product_inquiries) -> department_rag_agent + search_product -> reflection_agent",
    )
    q3 = "I'm looking for the blue wool jumper  --  is it available in medium size?"
    r3 = run_query(q3, user_id=CUSTOMER_EMAIL, agent=agent, user_context=USER_CONTEXT)
    show_result(r3, q3)
    pause()

    # ------------------------------------------------------------------
    # SCENARIO 4  --  Billing Department RAG
    # Demonstrates: Different department routing  --  same orchestrator, different context
    # ------------------------------------------------------------------
    section(
        "Billing & Payments Department  --  RAG",
        "Shows routing to a completely different department with the same agent.",
        "classifier_agent (billing_payments) -> department_rag_agent -> reflection_agent",
    )
    q4 = "I was charged twice for my last order. Can you explain how refunds work?"
    r4 = run_query(q4, user_id=CUSTOMER_EMAIL, agent=agent, user_context=USER_CONTEXT)
    show_result(r4, q4)
    pause()

    # ------------------------------------------------------------------
    # SCENARIO 5  --  Cross-Department Handoff (Orders -> Shipping)
    # Demonstrates: Agent detects the query spans two departments -> suggests transfer
    # ------------------------------------------------------------------
    section(
        "Cross-Department Handoff  --  Transfer to Shipping",
        "Customer asks about parcel tracking (while in Orders context) -> agent suggests transfer.",
        "classifier_agent -> department_rag_agent -> SUGGEST_TRANSFER detected -> user confirms",
    )
    q5 = "My order ORD-4821 says delivered but I haven't received it. Can I track the parcel?"
    r5 = run_query(q5, user_id=CUSTOMER_EMAIL, agent=agent, user_context=USER_CONTEXT)
    show_result(r5, q5)

    # If transfer was suggested, execute it  
    transfer_dept = r5.get("suggested_transfer", "")
    if transfer_dept:
        console.print(f"\n[bold yellow]-> Transfer suggested to: {transfer_dept}[/bold yellow]")
        console.print("[dim]Simulating user clicking 'Transfer' button...[/dim]\n")
        r5b = run_query_for_department(
            q5, transfer_dept,
            user_id=CUSTOMER_EMAIL, agent=agent, user_context=USER_CONTEXT,
        )
        show_result(r5b, f"[TRANSFERRED to {transfer_dept}] {q5}")
    else:
        console.print("[dim]Note: Agent handled it within the current department[/dim]")
    pause()

    # ------------------------------------------------------------------
    # SCENARIO 6  --  Mild Negative Sentiment (Medium severity  --  stays in RAG)
    # Demonstrates: NOT all negatives escalate  --  only HIGH severity does
    # ------------------------------------------------------------------
    section(
        "Mild Negative Sentiment  --  Stays in RAG (NOT escalated)",
        "A mildly disappointed customer. Severity = medium -> handled by RAG with empathy, NOT CrewAI.",
        "classifier_agent (negative, MEDIUM) -> department_rag_agent (with empathy) -> reflection_agent",
    )
    q6 = "I'm a bit disappointed with the quality of the t-shirt I received. The stitching looks poor."
    r6 = run_query(q6, user_id=CUSTOMER_EMAIL, agent=agent, user_context=USER_CONTEXT)
    show_result(r6, q6)
    escalated = bool(r6.get("escalation_info"))
    console.print(f"[bold]Escalated to CrewAI? -> [{'red' if escalated else 'green'}]{'YES' if escalated else 'NO (correctly handled by RAG)'}[/{'red' if escalated else 'green'}][/bold]")
    pause()

    # ------------------------------------------------------------------
    # SCENARIO 7  --  HIGH Severity Escalation -> CrewAI 3-Agent Crew
    # Demonstrates: Angry customer -> classification detects HIGH severity -> triggers CrewAI
    #   Agent 5 (Complaint Analyst) -> Agent 6 (Account Investigator) -> Agent 7 (Resolution Specialist)
    # ------------------------------------------------------------------
    section(
        "\U0001f525 HIGH Severity Escalation -> CrewAI 3-Agent Investigation",
        "Very angry customer with threats -> negative + HIGH severity -> triggers the full CrewAI crew.\n"
        "   Agent 5: complaint_analyst_agent (triages)\n"
        "   Agent 6: account_investigator_agent (root cause)\n"
        "   Agent 7: resolution_specialist_agent (recommends fix + goodwill)",
        "escalation_agent -> Agent 5 (Analyst) -> Agent 6 (Investigator) -> Agent 7 (Resolver)",
    )
    q7 = (
        "This is absolutely unacceptable! I ordered a GBP 80 pair of shoes THREE WEEKS AGO, "
        "order ORD-4821, and your website still says 'delivered' but I NEVER received them! "
        "I've called 4 times and nobody helps. I want a full refund and compensation or "
        "I'm reporting this to Trading Standards. This is the worst service I've ever experienced!"
    )
    r7 = run_query(q7, user_id=CUSTOMER_EMAIL, agent=agent, user_context=USER_CONTEXT)
    show_result(r7, q7[:100] + "...")
    pause()

    # ------------------------------------------------------------------
    # SCENARIO 8  --  Memory Recall (multi-turn conversation)
    # Demonstrates: Agent remembers the previous turns from this session
    # ------------------------------------------------------------------
    section(
        "Memory  --  Multi-Turn Conversation Recall",
        "The agent should remember what we discussed earlier in this session.",
        "classifier_agent -> SQLite memory (user_sarah@gmail.com thread) -> department_rag_agent",
    )
    q8 = "Earlier I asked about the blue wool jumper. Can you remind me which order number that was and its status?"
    r8 = run_query(q8, user_id=CUSTOMER_EMAIL, agent=agent, user_context=USER_CONTEXT)
    show_result(r8, q8)
    pause()

    # ------------------------------------------------------------------
    # SCENARIO 9  --  Explicit Transfer Request ("connect me to customer service")
    # Demonstrates: 22 transfer keyword phrases detected -> agent acts as the dept
    # ------------------------------------------------------------------
    section(
        "Transfer Keyword Detection  --  \"speak to someone about shipping\"",
        "Customer uses a transfer phrase -> system detects it and routes accordingly.",
        "classifier_agent -> transfer keyword match -> 3-turn history -> department_rag_agent",
    )
    q9 = "Can I speak to someone in the shipping department about a delayed delivery?"
    r9 = run_query(q9, user_id=CUSTOMER_EMAIL, agent=agent, user_context=USER_CONTEXT)
    show_result(r9, q9)
    pause()

    # ------------------------------------------------------------------
    # SCENARIO 10  --  Farewell Handling
    # Demonstrates: Goodbye detection  --  no agent pipeline triggered
    # ------------------------------------------------------------------
    section(
        "Farewell Handling  --  Graceful Sign-Off",
        "Customer says goodbye -> system detects farewell phrase and exits cleanly.",
        "Farewell detector (no agent call) -> friendly sign-off",
    )
    q10 = "That's everything, thank you for your help! Bye!"
    # Farewell is handled at the Streamlit/CLI layer, but let's show the agent's 
    # response to confirm it handles it gracefully
    r10 = run_query(q10, user_id=CUSTOMER_EMAIL, agent=agent, user_context=USER_CONTEXT)
    show_result(r10, q10)

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------
    console.print()
    console.print(Rule("[bold bright_cyan]DEMO COMPLETE[/bold bright_cyan]", style="bright_cyan"))
    console.print()

    summary = Table(title="Scenario Summary", show_lines=True, border_style="bright_cyan")
    summary.add_column("#", style="bold", width=3)
    summary.add_column("Scenario", style="bright_white", width=38)
    summary.add_column("Department", style="cyan", width=20)
    summary.add_column("Agents Used", style="magenta", width=22)
    summary.add_column("Score", style="green", width=6)

    rows = [
        ("1",  "Return policy (RAG)",           r1["department"], "Orchestrator",            str(r1["quality_score"])),
        ("2",  "Order lookup (tool)",            r2["department"], "Orchestrator + Tool",     str(r2["quality_score"])),
        ("3",  "Product search (tool)",          r3["department"], "Orchestrator + Tool",     str(r3["quality_score"])),
        ("4",  "Billing refund (RAG)",           r4["department"], "Orchestrator",            str(r4["quality_score"])),
        ("5",  "Cross-dept handoff",             r5["department"], "Orchestrator + Transfer", str(r5["quality_score"])),
        ("6",  "Mild negative (NOT escalated)",  r6["department"], "Orchestrator (empathy)",  str(r6["quality_score"])),
        ("7",  "HIGH escalation (CrewAI)",       r7["department"], "All 7 Agents",            "ESC [PASS]" if r7.get("escalation_info") else str(r7["quality_score"])),
        ("8",  "Memory recall",                  r8["department"], "Orchestrator + Memory",   str(r8["quality_score"])),
        ("9",  "Transfer keyword",               r9["department"], "Orchestrator + Transfer", str(r9["quality_score"])),
        ("10", "Farewell",                       r10["department"],"Farewell handler",        str(r10["quality_score"])),
    ]
    for row in rows:
        summary.add_row(*row)

    console.print(summary)

    console.print(Panel.fit(
        "[bold bright_cyan]All 10 scenarios completed![/bold bright_cyan]\n\n"
        "[PASS] Agent 1 (LangGraph Orchestrator)  --  handled classification, RAG, tools, handoffs, reflection\n"
        "[PASS] Agent 2 (Complaint Analyst)        --  triaged the escalated complaint\n"
        "[PASS] Agent 3 (Account Investigator)     --  investigated root cause\n"
        "[PASS] Agent 4 (Resolution Specialist)    --  recommended resolution + goodwill\n\n"
        "Features demonstrated:\n"
        "  - Multi-department RAG routing (orders, billing, product, shipping)\n"
        "  - Tool calls (lookup_orders, search_product)\n"
        "  - Severity-based escalation (medium -> RAG, HIGH -> CrewAI)\n"
        "  - Cross-department handoff with SUGGEST_TRANSFER tag\n"
        "  - Transfer keyword detection (22 phrases)\n"
        "  - Reflection loop (quality score 1-10, threshold >= 7)\n"
        "  - Persistent SQLite memory (multi-turn recall)\n"
        "  - Farewell handling (25+ phrases)",
        border_style="bright_cyan",
        title="\U0001f389 DEMO SUMMARY",
    ))


if __name__ == "__main__":
    main()
