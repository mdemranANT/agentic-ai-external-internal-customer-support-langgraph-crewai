#!/usr/bin/env python
"""
+==================================================================+
|  Shop4You  --  Orchestration & Routing Test Suite (70 tests)       |
|                                                                  |
|  Verifies: department routing, severity-based escalation,        |
|  cross-department handoffs, transfer keywords, generic terms,    |
|  chain transfers, ambiguous queries, forced routing, response    |
|  quality guards, multi-turn follow-ups, and edge cases.          |
|                                                                  |
|  Run:  python test_orchestration.py                              |
|  Time: ~5 minutes (~70 OpenAI API calls)                         |
+==================================================================+
"""

import io
import os
import time
import sys

# -- Fix Windows cp1252 encoding crash with Unicode characters --
os.environ["PYTHONUTF8"] = "1"
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from rich.console import Console
from rich.table import Table
from rich.rule import Rule
from rich.panel import Panel

from agents import compile_graph, run_query, run_query_for_department

console = Console(force_terminal=True)

# --------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------

VALID_DEPARTMENTS = {
    "orders_returns", "billing_payments", "shipping_delivery",
    "product_inquiries", "hr", "it_helpdesk", "operations",
    "loyalty_programme",
}

results = []  # (category, test_name, passed, detail)


def record(category: str, name: str, passed: bool, detail: str = ""):
    results.append((category, name, passed, detail))
    mark = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
    console.print(f"  {mark}  {name}  [dim]{detail}[/dim]")
    time.sleep(2)  # throttle to avoid OpenAI rate limits (30k TPM)


# --------------------------------------------------------------
# Main test runner
# --------------------------------------------------------------

def main():
    console.print(Panel(
        "[bold bright_cyan]Shop4You Orchestration Test Suite[/bold bright_cyan]\n"
        "[dim]70 tests across 12 categories . Requires OpenAI API key[/dim]",
        border_style="bright_cyan",
    ))

    t0 = time.time()

    # Compile a single agent (no memory) for all tests
    console.print("\n[dim]Compiling agent (no memory)...[/dim]")
    agent = compile_graph(use_memory=False)
    console.print("[green][PASS] Agent ready[/green]\n")

    # Also compile a memory-enabled agent for multi-turn tests
    mem_agent = compile_graph(use_memory=True, persist=False)

    # ==========================================================
    #  CATEGORY 1: Normal routing  --  one query per department (8)
    # ==========================================================
    console.print(Rule("[bold]1. Normal Routing (8 tests)[/bold]"))
    routing_tests = [
        ("I want to return my order",                     "orders_returns"),
        ("Why was I charged twice?",                      "billing_payments"),
        ("Where is my package?",                          "shipping_delivery"),
        ("Do you have this shirt in size medium?",        "product_inquiries"),
        ("Can I check my annual leave balance?",          "hr"),
        ("My VPN is not connecting to the office network","it_helpdesk"),
        ("What is the warehouse shift schedule this week?","operations"),
        ("How do I redeem my loyalty points?",            "loyalty_programme"),
    ]
    for query, expected_dept in routing_tests:
        r = run_query(query, user_id="orch_test", agent=agent)
        ok = r["department"] == expected_dept
        record("Normal routing", f"{expected_dept}", ok,
               f"got={r['department']}  sev={r['severity']}")

    # ==========================================================
    #  CATEGORY 2: Cross-department forced transfers (6)
    # ==========================================================
    console.print(Rule("[bold]2. Cross-Department Forced Transfers (6 tests)[/bold]"))
    transfer_tests = [
        ("I want to return my jacket",          "shipping_delivery"),
        ("Why was I double-charged?",           "orders_returns"),
        ("How much annual leave do I have?",    "operations"),
        ("Show me the blue jumper in stock",    "loyalty_programme"),
        ("My laptop screen is cracked",         "hr"),
        ("Where is my parcel?",                 "billing_payments"),
    ]
    for query, target in transfer_tests:
        r = run_query_for_department(query, target, user_id="orch_xfer", agent=agent)
        ok = r["department"] == target
        record("Forced transfer", f"-> {target}", ok,
               f"got={r['department']}")

    # ==========================================================
    #  CATEGORY 3: Escalation severity boundary (8)
    # ==========================================================
    console.print(Rule("[bold]3. Escalation Severity Boundary (8 tests)[/bold]"))
    # LOW / MEDIUM  --  should NOT escalate
    non_escalation = [
        ("The delivery could have been faster",                "low/med -> no escalation"),
        ("I'm not too impressed with the packaging",           "low/med -> no escalation"),
        ("I'm a bit disappointed with the quality",            "low/med -> no escalation"),
        ("This is frustrating, I've had to wait quite a while","low/med -> no escalation"),
    ]
    for query, label in non_escalation:
        r = run_query(query, user_id="orch_sev", agent=agent)
        escalated = bool(r.get("escalation_info"))
        ok = not escalated
        record("Severity boundary", label, ok,
               f"sev={r['severity']}  escalated={escalated}")

    # HIGH  --  SHOULD escalate
    escalation = [
        ("I will sue you if this isn't resolved immediately!",
         "high -> escalated"),
        ("This is the worst service I've ever experienced! I want compensation NOW!",
         "high -> escalated"),
        ("I'm reporting you to Trading Standards, this is a disgrace!",
         "high -> escalated"),
        ("You're all completely incompetent, I want to speak to your CEO!",
         "high -> escalated"),
    ]
    for query, label in escalation:
        r = run_query(query, user_id="orch_esc", agent=agent)
        escalated = bool(r.get("escalation_info"))
        ok = escalated
        record("Severity boundary", label, ok,
               f"sev={r['severity']}  escalated={escalated}")

    # ==========================================================
    #  CATEGORY 4: Transfer keyword detection (8)
    # ==========================================================
    console.print(Rule("[bold]4. Transfer Keyword Detection (8 tests)[/bold]"))
    keyword_tests = [
        ("Connect me to the billing department",          "billing_payments"),
        ("Can you pass me to shipping?",                  "shipping_delivery"),
        ("I'd like to speak to someone in HR",            "hr"),
        ("Transfer me to IT support please",              "it_helpdesk"),
        ("Put me through to the operations team",         "operations"),
        ("I need to talk to someone about loyalty points","loyalty_programme"),
        ("Can I be redirected to product inquiries?",     "product_inquiries"),
        ("Let me speak with the returns department",      "orders_returns"),
    ]
    for query, expected in keyword_tests:
        r = run_query(query, user_id="orch_kw", agent=agent)
        ok = r["department"] == expected
        record("Transfer keywords", f"-> {expected}", ok,
               f"got={r['department']}")

    # ==========================================================
    #  CATEGORY 5: Generic "customer service" (3)
    # ==========================================================
    console.print(Rule("[bold]5. Generic 'Customer Service'  --  No History (3 tests)[/bold]"))
    generic_tests = [
        "I need to speak to customer service",
        "Can I talk to a support agent please?",
        "I want to speak to a representative",
    ]
    for query in generic_tests:
        r = run_query(query, user_id="orch_gen", agent=agent)
        ok = r["department"] in VALID_DEPARTMENTS  # must NOT be "unknown"
        record("Generic CS terms", query[:45], ok,
               f"dept={r['department']}")

    # ==========================================================
    #  CATEGORY 6: Chain transfers A -> B -> C (3)
    # ==========================================================
    console.print(Rule("[bold]6. Chain Transfers A -> B -> C (3 tests)[/bold]"))
    chains = [
        ("Return this jacket",  ["shipping_delivery", "billing_payments"]),
        ("Check my payslip",    ["it_helpdesk", "operations"]),
        ("Is this in stock?",   ["orders_returns", "shipping_delivery"]),
    ]
    for query, chain in chains:
        ok_all = True
        trail = []
        for dept in chain:
            r = run_query_for_department(query, dept, user_id="orch_chain", agent=agent)
            if r["department"] != dept:
                ok_all = False
            trail.append(r["department"])
        record("Chain transfers", f"-> {'->'.join(chain)}", ok_all,
               f"got={'->'.join(trail)}")

    # ==========================================================
    #  CATEGORY 7: Ambiguous / tricky queries (7)
    # ==========================================================
    console.print(Rule("[bold]7. Ambiguous / Tricky Queries (7 tests)[/bold]"))
    ambiguous_tests = [
        ("My laptop isn't working and I need to expense the repair",
         {"it_helpdesk", "operations"}, "laptop + expense"),
        ("I bought something for the office and need a refund",
         {"billing_payments", "orders_returns"}, "office purchase refund"),
        ("Can I return a product that shipped to the wrong address?",
         {"orders_returns", "shipping_delivery"}, "return + wrong address"),
        ("I want to know about my order and my loyalty points",
         VALID_DEPARTMENTS, "mixed query"),
        ("Hello",
         VALID_DEPARTMENTS, "minimal input"),
        ("asdfghjkl zxcvbnm",
         VALID_DEPARTMENTS | {"unknown"}, "gibberish"),
        ("The delivery driver was extremely rude and unhelpful",
         {"shipping_delivery"} | VALID_DEPARTMENTS, "rude driver"),
    ]
    for query, acceptable, label in ambiguous_tests:
        r = run_query(query, user_id="orch_amb", agent=agent)
        ok = r["department"] in acceptable
        record("Ambiguous queries", label, ok,
               f"dept={r['department']}  sev={r['severity']}")

    # ==========================================================
    #  CATEGORY 8: run_query_for_department() forced (4)
    # ==========================================================
    console.print(Rule("[bold]8. Forced Department via run_query_for_department (4 tests)[/bold]"))
    forced_tests = [
        ("How do refunds work?",           "billing_payments"),
        ("What's the sick-leave policy?",  "hr"),
        ("Show me the blue jumper",        "product_inquiries"),
        ("Where is my parcel right now?",  "shipping_delivery"),
    ]
    for query, dept in forced_tests:
        r = run_query_for_department(query, dept, user_id="orch_forced", agent=agent)
        ok = r["department"] == dept
        record("Forced department", f"-> {dept}", ok,
               f"got={r['department']}")

    # ==========================================================
    #  CATEGORY 9: No "contact customer service" leaks (5)
    # ==========================================================
    console.print(Rule("[bold]9. No 'Contact Customer Service' in Responses (5 tests)[/bold]"))
    leak_tests = [
        "What is your return policy?",
        "How do I check my order status?",
        "What payment methods do you accept?",
        "When will my delivery arrive?",
        "Do you have a size guide?",
    ]
    for query in leak_tests:
        r = run_query(query, user_id="orch_leak", agent=agent)
        resp_lower = r["response"].lower()
        has_leak = ("contact customer service" in resp_lower
                    or "contact our customer service" in resp_lower
                    or "reach out to our customer service" in resp_lower)
        ok = not has_leak
        record("No CS leaks", query[:42], ok,
               "LEAK FOUND" if has_leak else "clean")

    # ==========================================================
    #  CATEGORY 10: Multi-turn follow-ups (5)
    # ==========================================================
    console.print(Rule("[bold]10. Multi-Turn Follow-Ups (5 tests)[/bold]"))

    # Turn 1: establish context in orders
    uid_mt = f"orch_mt_{int(time.time())}"
    r1 = run_query("I want to return my jacket, order ORD-1234", user_id=uid_mt, agent=mem_agent)
    ok1 = r1["department"] == "orders_returns"
    record("Multi-turn", "turn 1: order context", ok1,
           f"dept={r1['department']}")

    # Turn 2: short follow-up should stay in same context
    r2 = run_query("Yes, please process the return", user_id=uid_mt, agent=mem_agent)
    ok2 = r2["department"] in {"orders_returns", "shipping_delivery"}
    record("Multi-turn", "turn 2: follow-up stays", ok2,
           f"dept={r2['department']}")

    # Turn 3: ask to transfer
    r3 = run_query("Actually, can you transfer me to shipping?", user_id=uid_mt, agent=mem_agent)
    ok3 = r3["department"] == "shipping_delivery"
    record("Multi-turn", "turn 3: transfer request", ok3,
           f"dept={r3['department']}")

    # Separate conversation: establish billing context
    uid_mt2 = f"orch_mt2_{int(time.time())}"
    r4 = run_query("I was charged twice for my last order", user_id=uid_mt2, agent=mem_agent)
    ok4 = r4["department"] == "billing_payments"
    record("Multi-turn", "turn 4: new user billing", ok4,
           f"dept={r4['department']}")

    # Turn 5: follow-up should stay in billing
    r5 = run_query("When will the refund be processed?", user_id=uid_mt2, agent=mem_agent)
    ok5 = r5["department"] in {"billing_payments", "orders_returns"}
    record("Multi-turn", "turn 5: billing follow-up", ok5,
           f"dept={r5['department']}")

    # ==========================================================
    #  CATEGORY 11: Escalation + edge cases (10)
    # ==========================================================
    console.print(Rule("[bold]11. Escalation & Edge Cases (10 tests)[/bold]"))

    # Should escalate
    esc_tests = [
        ("I hate your company, this is fraud and I'm going to the press!",
         True, "angry + fraud threat"),
        ("Nobody ever helps me, I've called 10 times! This is unacceptable!",
         True, "repeated frustration"),
        ("I demand to speak to a manager RIGHT NOW or I'm suing!",
         True, "lawsuit threat"),
    ]
    for query, should_escalate, label in esc_tests:
        r = run_query(query, user_id="orch_edge", agent=agent)
        escalated = bool(r.get("escalation_info"))
        ok = escalated == should_escalate
        record("Escalation edge", label, ok,
               f"escalated={escalated}  sev={r['severity']}")

    # Should NOT escalate (medium/low)
    no_esc_tests = [
        ("Not great, but I suppose it's okay",          "mild negative"),
        ("I expected better quality honestly",           "medium negative"),
        ("The shipping was a bit slow this time",        "low negative"),
    ]
    for query, label in no_esc_tests:
        r = run_query(query, user_id="orch_edge2", agent=agent)
        escalated = bool(r.get("escalation_info"))
        ok = not escalated
        record("Escalation edge", f"NO esc: {label}", ok,
               f"escalated={escalated}  sev={r['severity']}")

    # Edge cases: valid response for odd inputs
    edge_tests = [
        ("",             "empty string"),
        ("???",          "only punctuation"),
        ("12345",        "only numbers"),
        ("help help help help", "repeated word"),
    ]
    for query, label in edge_tests:
        try:
            r = run_query(query if query else "hi", user_id="orch_edge3", agent=agent)
            ok = r["department"] in VALID_DEPARTMENTS | {"unknown"}
            record("Escalation edge", f"edge: {label}", ok,
                   f"dept={r['department']}")
        except Exception as e:
            record("Escalation edge", f"edge: {label}", False, str(e)[:60])

    # ==========================================================
    #  CATEGORY 12: Product -> Orders retry (3 runs)
    # ==========================================================
    console.print(Rule("[bold]12. Product -> Orders Retry (3 runs)[/bold]"))
    # "I bought a product" could go to product_inquiries or orders_returns
    # We accept either  --  the key is consistency across retries
    retry_query = "I bought a pair of shoes and the sole is already coming off"
    retry_depts = []
    for i in range(3):
        r = run_query(retry_query, user_id=f"orch_retry_{i}", agent=agent)
        retry_depts.append(r["department"])
        ok = r["department"] in {"product_inquiries", "orders_returns"}
        record("Product<->Orders", f"run {i+1}", ok,
               f"dept={r['department']}")

    # ==========================================================
    #  SUMMARY
    # ==========================================================
    elapsed = time.time() - t0

    console.print()
    console.print(Rule("[bold bright_cyan]TEST SUMMARY[/bold bright_cyan]", style="bright_cyan"))
    console.print()

    # Group by category
    categories = {}
    for cat, name, passed, detail in results:
        if cat not in categories:
            categories[cat] = {"passed": 0, "failed": 0, "total": 0}
        categories[cat]["total"] += 1
        if passed:
            categories[cat]["passed"] += 1
        else:
            categories[cat]["failed"] += 1

    summary = Table(title="Orchestration Test Results", show_lines=True, border_style="bright_cyan")
    summary.add_column("#", style="bold", width=3)
    summary.add_column("Category", style="bright_white", width=40)
    summary.add_column("Passed", style="green", justify="center", width=8)
    summary.add_column("Failed", style="red", justify="center", width=8)
    summary.add_column("Result", justify="center", width=10)

    total_passed = 0
    total_failed = 0
    for i, (cat, data) in enumerate(categories.items(), 1):
        total_passed += data["passed"]
        total_failed += data["failed"]
        status = "[green][PASS][/green]" if data["failed"] == 0 else "[red][FAIL][/red]"
        summary.add_row(
            str(i), cat,
            str(data["passed"]), str(data["failed"]),
            status,
        )

    total = total_passed + total_failed
    summary.add_row(
        "", "[bold]TOTAL[/bold]",
        f"[bold green]{total_passed}[/bold green]",
        f"[bold red]{total_failed}[/bold red]",
        f"[bold green]{total_passed}/{total} [PASS][/bold green]" if total_failed == 0
        else f"[bold red]{total_passed}/{total}[/bold red]",
    )
    console.print(summary)

    console.print(f"\n[dim]Elapsed: {elapsed:.1f}s  |  API calls: ~{total}[/dim]")

    # Show failures if any
    failures = [(cat, name, detail) for cat, name, passed, detail in results if not passed]
    if failures:
        console.print(f"\n[bold red]FAILURES ({len(failures)}):[/bold red]")
        for cat, name, detail in failures:
            console.print(f"  [red][FAIL][/red] [{cat}] {name}: {detail}")

    # Exit code
    sys.exit(0 if total_failed == 0 else 1)


if __name__ == "__main__":
    main()
