"""
app_cli.py
----------
Terminal chat interface for the Power Systems RAG assistant.

Usage:
    python app_cli.py
"""
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from rag_engine import PowerSystemsRAG, ConversationState

console = Console()


def main():
    console.print(
        Panel.fit(
            "[bold cyan]Power Systems Engineering RAG Assistant[/bold cyan]\n"
            "Ask about per-unit systems, load flow, fault analysis, protection, "
            "stability, transformers, and more.\nType 'exit' to quit, 'reset' to clear memory.",
            border_style="cyan",
        )
    )

    try:
        rag = PowerSystemsRAG()
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        return
    except Exception as e:
        console.print(f"[red]Unexpected error connecting to Ollama: {e}[/red]")
        return

    state = ConversationState()

    while True:
        try:
            question = console.input("\n[bold green]You:[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Goodbye![/yellow]")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            console.print("[yellow]Goodbye![/yellow]")
            break
        if question.lower() == "reset":
            state = ConversationState()
            console.print("[yellow]Conversation memory cleared.[/yellow]")
            continue

        with console.status("[cyan]Retrieving context & thinking...[/cyan]"):
            result = rag.ask(question, state)

        console.print("\n[bold magenta]Assistant:[/bold magenta]")
        console.print(Markdown(result["answer"]))

        if result["sources"]:
            src_lines = "\n".join(
                f"  - {s['source']} (chunk {s['chunk_index']}, relevance {s['relevance']})"
                for s in result["sources"]
            )
            console.print(f"[dim]Sources used:\n{src_lines}[/dim]")


if __name__ == "__main__":
    main()
