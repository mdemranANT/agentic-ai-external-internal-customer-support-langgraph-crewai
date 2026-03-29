"""
Standalone RAG pipeline tests.
Checks retrieval accuracy and response generation for every department,
plus a few edge cases (ambiguous queries, empty results, cross-department).

Usage:  python test_rag.py
"""
import time
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from config import DEPARTMENTS, OPENAI_API_KEY, LLM_MODEL, LLM_TEMPERATURE
from vector_store import create_vector_store, retrieve_context, get_retriever
from prompts import RAG_PROMPT, get_rag_prompt_vars, NO_CONTEXT_FALLBACK
from langchain_openai import ChatOpenAI

console = Console()

# LLM for generation tests---
llm = ChatOpenAI(
    model=LLM_MODEL,
    temperature=LLM_TEMPERATURE,
    openai_api_key=OPENAI_API_KEY,
)


# ---------- Test queries  --  one per department + edge cases ----------
DEPARTMENT_TEST_QUERIES = {
    "orders_returns": "How do I return a faulty item I received?",
    "billing_payments": "What payment methods do you accept?",
    "shipping_delivery": "How long does standard delivery take in the UK?",
    "product_inquiries": "Do you have a size guide for shoes?",
    "hr": "How do I apply for annual leave?",
    "it_helpdesk": "How do I reset my password?",
    "operations": "What is the warehouse shift schedule?",
    "loyalty_programme": "How do I check my loyalty points balance?",
}

EDGE_CASE_QUERIES = [
    ("Ambiguous query", "I need help", None),
    ("Empty retrieval", "quantum physics equations for string theory", "orders_returns"),
    ("Cross-department", "I want to return an item and check my loyalty points", "orders_returns"),
    ("Very short query", "refund", "billing_payments"),
    ("Greeting", "Hello, how are you?", None),
]


def test_retrieval_per_department(vs):
    """Makes sure each department actually returns docs that belong to it."""
    console.print(Panel("[bold]Test 1: Retrieval per Department[/bold]", style="cyan"))

    table = Table(title="Department Retrieval Results")
    table.add_column("Department", style="bold")
    table.add_column("Query", max_width=40)
    table.add_column("Docs", justify="center")
    table.add_column("Correct Dept?", justify="center")
    table.add_column("Preview", max_width=50)

    passed = 0
    total = len(DEPARTMENT_TEST_QUERIES)

    for dept_key, query in DEPARTMENT_TEST_QUERIES.items():
        context, docs = retrieve_context(vs, query, dept_key)
        num_docs = len(docs)
        dept_name = DEPARTMENTS[dept_key]["name"]

    # Make sure retrieved docs actually come from the right department
        correct = all(
            d.metadata.get("department_key") == dept_key for d in docs
        ) if docs else False

        status = "[green][PASS][/green]" if (correct and num_docs > 0) else "[red][FAIL][/red]"
        if correct and num_docs > 0:
            passed += 1

        preview = context[:80].replace("\n", " ") + "..." if context else "[dim]No results[/dim]"

        table.add_row(dept_name, query, str(num_docs), status, preview)

    console.print(table)
    console.print(f"  Result: {passed}/{total} departments retrieved correctly\n")
    return passed == total


def test_fallback_retrieval(vs):
    """Checks that the global fallback kicks in when a dept-filtered search finds nothing."""
    console.print(Panel("[bold]Test 2: Fallback Retrieval[/bold]", style="cyan"))

    # A returns query filtered to IT shouldn't match, but global might
    query = "What is the return policy for electronics?"
    context_filtered, docs_filtered = retrieve_context(vs, query, "it_helpdesk", fallback_to_global=False)
    context_fallback, docs_fallback = retrieve_context(vs, query, "it_helpdesk", fallback_to_global=True)

    console.print(f"  Query: '{query}' (filtered to it_helpdesk)")
    console.print(f"  Without fallback: {len(docs_filtered)} doc(s)")
    console.print(f"  With fallback:    {len(docs_fallback)} doc(s)")

    if len(docs_fallback) >= len(docs_filtered):
        console.print("  [green][PASS] Fallback retrieval works as expected[/green]\n")
        return True
    else:
        console.print("  [red][FAIL] Fallback retrieval did not improve results[/red]\n")
        return False


def test_rag_generation(vs):
    """Full retrieve-then-generate cycle for every department."""
    console.print(Panel("[bold]Test 3: RAG Generation (retrieve + generate)[/bold]", style="cyan"))

    table = Table(title="RAG Generation Results")
    table.add_column("Department", style="bold")
    table.add_column("Query", max_width=35)
    table.add_column("Time", justify="right")
    table.add_column("Response Preview", max_width=55)

    for dept_key, query in DEPARTMENT_TEST_QUERIES.items():
        dept_name = DEPARTMENTS[dept_key]["name"]
        start = time.time()

        # Retrieve
        context, _ = retrieve_context(vs, query, dept_key)

        # Generate using RAG prompt template
        prompt_vars = get_rag_prompt_vars(dept_key, query, context)
        formatted = RAG_PROMPT.format_messages(**prompt_vars)
        response = llm.invoke(formatted)

        elapsed = time.time() - start
        preview = response.content[:100].replace("\n", " ") + "..."

        table.add_row(dept_name, query, f"{elapsed:.1f}s", preview)

    console.print(table)
    console.print()
    return True


def test_edge_cases(vs):
    """Throws some tricky queries at the retriever to see how it copes."""
    console.print(Panel("[bold]Test 4: Edge Cases[/bold]", style="cyan"))

    table = Table(title="Edge Case Results")
    table.add_column("Case", style="bold")
    table.add_column("Query", max_width=40)
    table.add_column("Dept Filter")
    table.add_column("Docs", justify="center")
    table.add_column("Has Context?", justify="center")

    for case_name, query, dept_key in EDGE_CASE_QUERIES:
        context, docs = retrieve_context(vs, query, dept_key)
        has_ctx = "[green]Yes[/green]" if context else "[yellow]No[/yellow]"
        dept_label = dept_key if dept_key else "[dim]None[/dim]"
        table.add_row(case_name, query, dept_label, str(len(docs)), has_ctx)

    console.print(table)
    console.print()
    return True


def test_metadata_isolation(vs):
    """Confirms that HR docs don't bleed into Shipping results and vice-versa."""
    console.print(Panel("[bold]Test 5: Metadata Isolation[/bold]", style="cyan"))

    # Try an HR question on the Shipping collection
    query = "How do I apply for annual leave?"
    context, docs = retrieve_context(vs, query, "shipping_delivery", fallback_to_global=False)

    if not docs:
        console.print("  [green][PASS] No HR content leaked into Shipping department[/green]")
        isolated = True
    else:
        leaked = any(d.metadata.get("department_key") != "shipping_delivery" for d in docs)
        if leaked:
            console.print("  [red][FAIL] Cross-department leakage detected![/red]")
            isolated = False
        else:
            console.print("  [green][PASS] Results are all from Shipping (topic overlap possible)[/green]")
            isolated = True

    # And the reverse: a Shipping question aimed at HR
    query2 = "How long does standard delivery take?"
    context2, docs2 = retrieve_context(vs, query2, "hr", fallback_to_global=False)
    if not docs2:
        console.print("  [green][PASS] No Shipping content leaked into HR department[/green]")
    else:
        console.print(f"  [yellow][!] {len(docs2)} doc(s) found in HR for shipping query (possible overlap)[/yellow]")

    console.print()
    return isolated


# ---------- Main ----------
if __name__ == "__main__":
    console.print(Panel.fit(
        "[bold blue]Shop4You  --  RAG Pipeline Tests[/bold blue]\n"
        "Testing retrieval, generation, edge cases, and isolation",
        border_style="blue",
    ))
    console.print()

    # Load existing vector store (don't re-ingest)
    console.print("[dim]Loading vector store...[/dim]")
    vs = create_vector_store()

    results = {}

    # Run all tests
    results["Retrieval per Dept"] = test_retrieval_per_department(vs)
    results["Fallback Retrieval"] = test_fallback_retrieval(vs)
    results["RAG Generation"] = test_rag_generation(vs)
    results["Edge Cases"] = test_edge_cases(vs)
    results["Metadata Isolation"] = test_metadata_isolation(vs)

    # Summary
    console.print(Panel("[bold]Test Summary[/bold]", style="green"))
    for test_name, passed in results.items():
        icon = "[green][PASS] PASS[/green]" if passed else "[red][FAIL] FAIL[/red]"
        console.print(f"  {icon}  {test_name}")

    total_pass = sum(results.values())
    total = len(results)
    console.print(f"\n  [bold]{total_pass}/{total} tests passed[/bold]")
