"""
Interactive CLI chat for the Shop4You agent.
Supports multi-user sessions backed by persistent SQLite memory.

Usage:  python main.py
"""
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from agents import compile_graph, run_query

console = Console()


def main():
    console.print(Panel.fit(
        "[bold blue]Shop4You AI Assistant[/bold blue]\n"
        "Ask about orders, billing, shipping, products, HR, IT, operations, or loyalty.\n"
        "Type [bold]quit[/bold] to exit.",
        border_style="blue",
    ))

    user_id = console.input("\n[bold]Enter your user ID (or press Enter for 'guest'): [/bold]").strip()
    if not user_id:
        user_id = "guest"

    console.print(f"[dim]Session started for user: {user_id}  (memory: persistent SQLite)[/dim]\n")

    # Persistent memory so the conversation sticks around between runs
    agent = compile_graph(use_memory=True, persist=True)

    while True:
        try:
            query = console.input("[bold green]You:[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break

        console.print("[dim]Thinking...[/dim]")

        try:
            result = run_query(query, user_id=user_id, agent=agent)

            # Show classification metadata
            dept = result.get("department", "unknown")
            sentiment = result.get("sentiment", "unknown")
            score = result.get("quality_score", " -- ")
            reasoning = result.get("classification_reasoning", "")
            console.print(
                f"[dim]  Department: {dept} | Sentiment: {sentiment} | "
                f"Quality: {score}/10[/dim]"
            )
            if reasoning:
                console.print(f"[dim]  Routing reason: {reasoning}[/dim]")

            # Show response
            response = result.get("response", "Sorry, something went wrong.")
            console.print(Panel(
                Markdown(response),
                title="[bold blue]Shop4You Assistant[/bold blue]",
                border_style="blue",
            ))

            # Show escalation info if applicable
            esc = result.get("escalation_info", {})
            if esc:
                ref = esc.get("reference_number", "N/A")
                console.print(
                    f"[yellow][!] Escalated to human agent   --   Ref: {ref}[/yellow]"
                )

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

        console.print()


if __name__ == "__main__":
    main()
