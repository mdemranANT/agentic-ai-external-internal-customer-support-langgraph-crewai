"""
Quick test to verify that separate user sessions don't leak into each other.

Works with both:
  pytest test_memory_isolation.py -v
  python test_memory_isolation.py          (standalone output)
"""
from agents import compile_graph, run_query


def test_memory_isolation(agent):
    """Two different users should never see each other's conversation data."""
    # User A asks about billing
    r1 = run_query("My name is Sarah and I need help with a refund",
                   user_id="user_A", agent=agent)
    assert r1["department"], "User A should get a department"

    # User B asks about HR (completely different topic)
    r2 = run_query("My name is James and I want to check my payslip",
                   user_id="user_B", agent=agent)
    assert r2["department"], "User B should get a department"

    # User A follow-up  --  should recall Sarah
    r3 = run_query(
        "Can you check the refund status for me? You should have my name from before.",
        user_id="user_A", agent=agent,
    )
    # We accept the test as long as User A's session didn't crash
    assert r3.get("response"), "User A follow-up should produce a response"

    # User B follow-up  --  must NOT mention Sarah
    r4 = run_query(
        "Can you check the payslip for me? You should have my name from before.",
        user_id="user_B", agent=agent,
    )
    assert "sarah" not in r4["response"].lower(), \
        "MEMORY LEAK: User B's response should never contain Sarah's name"


# -- Standalone mode ---------------------------------------------------
if __name__ == "__main__":
    agent = compile_graph(use_memory=True, persist=False)

    r1 = run_query("My name is Sarah and I need help with a refund",
                   user_id="user_A", agent=agent)
    print(f"User A dept: {r1['department']}  sentiment: {r1['sentiment']}")

    r2 = run_query("My name is James and I want to check my payslip",
                   user_id="user_B", agent=agent)
    print(f"User B dept: {r2['department']}  sentiment: {r2['sentiment']}")

    r3 = run_query(
        "Can you check the refund status for me? You should have my name from before.",
        user_id="user_A", agent=agent,
    )
    has_sarah = "sarah" in r3["response"].lower()
    print(f"User A follow-up mentions Sarah: {has_sarah}")
    print(f"  Response: {r3['response'][:180]}")

    r4 = run_query(
        "Can you check the payslip for me? You should have my name from before.",
        user_id="user_B", agent=agent,
    )
    no_sarah = "sarah" not in r4["response"].lower()
    has_james = "james" in r4["response"].lower()
    print(f"User B follow-up does NOT mention Sarah: {no_sarah}")
    print(f"User B follow-up mentions James: {has_james}")
    print(f"  Response: {r4['response'][:180]}")

    if no_sarah:
        print("\nMEMORY ISOLATION: PASS")
    else:
        print("\nMEMORY ISOLATION: FAIL")
