"""
ingest.py
---------
Command-line entry point for building the knowledge base from files placed in
data/. (You can also do this from inside the web app now — see app_streamlit.py,
which has a built-in file uploader.)

Usage:
    python ingest.py                 # ingest everything currently in data/
    python ingest.py --reset         # wipe the collection and re-ingest
"""
import argparse

import chromadb
from rich.console import Console

import config
import knowledge_base as kb

console = Console()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Wipe existing collection first")
    args = parser.parse_args()

    config.DATA_DIR.mkdir(exist_ok=True)
    config.DB_DIR.mkdir(exist_ok=True)

    if args.reset:
        client = chromadb.PersistentClient(path=str(config.DB_DIR))
        try:
            client.delete_collection(config.COLLECTION_NAME)
            console.log("[yellow]Existing collection deleted.[/yellow]")
        except Exception:
            pass

    console.log(f"Scanning {config.DATA_DIR} for documents...")
    summary = kb.ingest_all()

    if summary["files_processed"] == 0:
        console.log(
            f"[red]No documents found in {config.DATA_DIR}. "
            "Add PDFs/TXT/MD files there and re-run.[/red]"
        )
        return

    for filename, n in summary["per_file"].items():
        console.log(f"  {filename}: {n} chunks")

    console.log(
        f"[green]Done. Ingested {summary['files_processed']} documents / "
        f"{summary['total_chunks']} chunks into collection '{config.COLLECTION_NAME}'.[/green]"
    )


if __name__ == "__main__":
    main()
