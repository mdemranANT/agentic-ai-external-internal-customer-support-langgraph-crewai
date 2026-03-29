"""
End-to-end tests for the full agent graph.
Covers routing, RAG, escalation, product tools, and multi-turn memory.

Works with both:
  pytest test_agent.py -v
  python test_agent.py          (standalone Rich output)
"""
import time
import pytest
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agents import compile_graph, run_query

console = Console()


# -- Helper ------------------------------------------------------------
def _run_one(agent, query, user_id, expect_escalation, expect_dept):
    """Run a single E2E test and return (result_dict, passed_bool)."""
    result = run_query(query, user_id=user_id, agent=agent)
    dept = result.get("department", "?")
    is_escalated = bool(result.get("escalation_info"))

    ok = True
    if expect_escalation and not is_escalated:
        ok = False
    if not expect_escalation and is_escalated:
        ok = False
    if expect_dept and expect_dept != dept:
        ok = False
    return result, ok


# -- Pytest test functions (use 'agent' fixture from conftest.py) ------

def test_billing_rag(agent):
    """Normal RAG  --  billing department."""
    _, ok = _run_one(agent, "What payment methods do you accept?",
                     "test_billing", False, "billing_payments")
    assert ok, "Expected billing_payments department, no escalation"


def test_hr_rag(agent):
    """Normal RAG  --  HR internal department."""
    _, ok = _run_one(agent, "How do I apply for annual leave?",
                     "test_hr", False, "hr")
    assert ok, "Expected hr department, no escalation"


def test_product_tool(agent):
    """Product search tool (Stretch Goal 3)."""
    _, ok = _run_one(agent, "Do you have running shoes in stock?",
                     "test_product", False, "product_inquiries")
    assert ok, "Expected product_inquiries department, no escalation"


def test_negative_escalation(agent):
    """Negative sentiment -> CrewAI escalation."""
    _, ok = _run_one(
        agent,
        "This is absolutely terrible! I have been waiting 3 weeks and nobody helps me!",
        "test_angry", True, None,
    )
    assert ok, "Expected escalation for highly negative sentiment"


def test_unknown_dept_escalation(agent):
    """Unknown department -> escalation."""
    _, ok = _run_one(agent, "Can you help me file my taxes?",
                     "test_unknown", True, "unknown")
    assert ok, "Expected unknown department classification and escalation"


def test_memory_turn1(agent):
    """Multi-turn memory  --  turn 1 (store context)."""
    _, ok = _run_one(
        agent,
        "My name is Alice and I want to track my order ORD-12345",
        "test_memory", False, "orders_returns",
    )
    assert ok, "Expected orders_returns department, no escalation"


def test_memory_turn2(agent):
    """Multi-turn memory  --  turn 2 (recall from same user)."""
    result, _ = _run_one(
        agent,
        "What was the order number I just mentioned?",
        "test_memory", False, None,
    )
    # We mainly care that it didn't escalate and produced a response
    assert not bool(result.get("escalation_info")), "Turn 2 should not escalate"


# -- Standalone mode (python test_agent.py) ----------------------------

_ALL_TESTS = [
    {"name": "Normal RAG (billing)",
     "query": "What payment methods do you accept?",
     "user_id": "test_billing",
     "expect_escalation": False, "expect_dept": "billing_payments"},
    {"name": "Normal RAG (HR  --  internal)",
     "query": "How do I apply for annual leave?",
     "user_id": "test_hr",
     "expect_escalation": False, "expect_dept": "hr"},
    {"name": "Product search tool (Stretch Goal 3)",
     "query": "Do you have running shoes in stock?",
     "user_id": "test_product",
     "expect_escalation": False, "expect_dept": "product_inquiries"},
    {"name": "Negative sentiment -> Escalation",
     "query": "This is absolutely terrible! I have been waiting 3 weeks and nobody helps me!",
     "user_id": "test_angry",
     "expect_escalation": True, "expect_dept": None},
    {"name": "Unknown department -> Escalation",
     "query": "Can you help me file my taxes?",
     "user_id": "test_unknown",
     "expect_escalation": True, "expect_dept": "unknown"},
    {"name": "Multi-turn memory (turn 1)",
     "query": "My name is Alice and I want to track my order ORD-12345",
     "user_id": "test_memory",
     "expect_escalation": False, "expect_dept": "orders_returns"},
    {"name": "Multi-turn memory (turn 2  --  same user)",
     "query": "What was the order number I just mentioned?",
     "user_id": "test_memory",
     "expect_escalation": False, "expect_dept": None},
]


def main():
    console.print(Panel.fit(
        "[bold blue]Shop4You  --  Full Agent E2E Tests[/bold blue]",
        border_style="blue",
    ))

    agent = compile_graph(use_memory=True, persist=False)
    console.print("[dim]Agent compiled (in-memory mode)[/dim]\n")

    table = Table(title="Agent E2E Test Results")
    table.add_column("#", justify="center", width=3)
    table.add_column("Test", max_width=35)
    table.add_column("Dept", max_width=18)
    table.add_column("Sentiment")
    table.add_column("Quality", justify="center")
    table.add_column("Esc?", justify="center")
    table.add_column("Time", justify="right")
    table.add_column("Pass?", justify="center")

    passed = 0
    for i, t in enumerate(_ALL_TESTS, 1):
        console.print(f"[dim]Running test {i}/{len(_ALL_TESTS)}: {t['name']}...[/dim]")
        start = time.time()
        try:
            result, ok = _run_one(
                agent, t["query"], t["user_id"],
                t["expect_escalation"], t["expect_dept"],
            )
            elapsed = time.time() - start
            if ok:
                passed += 1
            table.add_row(
                str(i), t["name"],
                result.get("department", "?"),
                result.get("sentiment", "?"),
                str(result.get("quality_score", "?")),
                "[yellow]Yes[/yellow]" if bool(result.get("escalation_info")) else "No",
                f"{elapsed:.1f}s",
                "[green][PASS][/green]" if ok else "[red][FAIL][/red]",
            )
            preview = result.get("response", "")[:120].replace("\n", " ")
            console.print(f"  [dim]{preview}...[/dim]")
        except Exception as e:
            elapsed = time.time() - start
            table.add_row(str(i), t["name"], "ERR", "ERR", " -- ", " -- ", f"{elapsed:.1f}s", "[red][FAIL][/red]")
            console.print(f"  [red]Error: {e}[/red]")

    console.print()
    console.print(table)
    console.print(f"\n  [bold]{passed}/{len(_ALL_TESTS)} tests passed[/bold]")


if __name__ == "__main__":
    main()
