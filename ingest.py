# -*- coding: utf-8 -*-
"""Load support tickets and build or open their persisted Chroma index.

Preparation: run `python ingest.py` with OPENAI_API_KEY in the environment.
Runtime: use load_tickets() and load_vectorstore(), then pass both to tools.
Importing this module does not read ticket files or make API calls.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Union

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

BASE_DIRECTORY = Path(__file__).resolve().parent
TICKETS_PATH = BASE_DIRECTORY / "data" / "synthetic_tickets.json"
PERSIST_DIRECTORY = BASE_DIRECTORY / "agent_vectorstore"
# Preserve the default collection used by the original tools.py.
COLLECTION_NAME = "tickets"


def load_tickets(tickets_path: Union[str, Path] = TICKETS_PATH) -> List[Dict[str, Any]]:
    """Read the ticket JSON independently of the terminal working directory."""
    with Path(tickets_path).open(encoding="utf-8") as file:
        return json.load(file)


def create_documents(tickets: List[Dict[str, Any]]) -> List[Document]:
    """Preserve the original ticket text and metadata used for retrieval."""
    documents = []
    for ticket in tickets:
        content = f"""Ticket ID: {ticket['ticket_id']}
Title: {ticket['title']}
Description: {ticket['description']}
Resolution: {ticket['resolution']}
Category: {ticket['category']}
Priority: {ticket['priority']}"""
        
        doc = Document(
            page_content=content,
            metadata={
                'ticket_id': ticket['ticket_id'],
                'category': ticket['category'],
                'priority': ticket['priority'],
                'title': ticket['title']
            }
        )
        documents.append(doc)
    
    return documents


def create_embeddings() -> OpenAIEmbeddings:
    """Configure the embedding model shared by ingestion and query retrieval."""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY in your environment first.")
    return OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    )


def build_vectorstore(
    tickets: List[Dict[str, Any]],
    persist_directory: Union[str, Path] = PERSIST_DIRECTORY,
) -> Chroma:
    """Embed and upsert tickets. Call explicitly when preparing/updating data.

    Stable ticket IDs avoid duplicate entries on repeated ingestion. This does
    not delete tickets removed from the JSON. Use the same embedding model
    when building and opening an index.
    """
    documents = create_documents(tickets)
    if not documents:
        raise ValueError("No tickets to index.")
    ids = [str(ticket["ticket_id"]) for ticket in tickets]
    if len(ids) != len(set(ids)):
        raise ValueError("Ticket IDs must be unique.")
    return Chroma.from_documents(
        documents=documents,
        embedding=create_embeddings(),
        ids=ids,
        collection_name=COLLECTION_NAME,
        persist_directory=str(Path(persist_directory).resolve()),
    )


def load_vectorstore(persist_directory: Union[str, Path] = PERSIST_DIRECTORY) -> Chroma:
    """Open an existing index without re-embedding its documents.

    Semantic searches still call the embedding API to embed each query.
    """
    directory = Path(persist_directory).resolve()
    if not (directory / "chroma.sqlite3").is_file():
        raise FileNotFoundError("Chroma index not found. Run ingest.py first.")
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=create_embeddings(),
        persist_directory=str(directory),
        create_collection_if_not_exists=False,
    )


def main() -> None:
    tickets = load_tickets()
    build_vectorstore(tickets)
    print(f"Indexed {len(tickets)} support tickets in {PERSIST_DIRECTORY}")


if __name__ == "__main__":
    main()
