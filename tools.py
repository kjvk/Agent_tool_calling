# -*- coding: utf-8 -*-
"""Four runtime tools for the support-ticket agent.

Data loading and Chroma indexing live in ingest.py. This module only uses
resources supplied by the application; importing it performs no ingestion.
"""

from typing import List, Dict, Any

from langchain_core.tools import Tool
from langchain_chroma import Chroma


class SupportTicketTools:
    """Collection of tools backed by already-loaded tickets and Chroma."""

    def __init__(self, tickets: List[Dict[str, Any]], vectorstore: Chroma):
        """Receive dependencies; do not load files or create embeddings here.

        Example application setup (after running ingest.py once):
            from ingest import load_tickets, load_vectorstore
            ticket_tools = SupportTicketTools(load_tickets(), load_vectorstore())
            tools = ticket_tools.get_tools()
        """
        self.tickets = tickets
        self.vectorstore = vectorstore

    def search_similar_tickets(self, query: str) -> str:
        """
        Search for similar support tickets using semantic similarity.
        
        Args:
            query: User's problem description or question
            
        Returns:
            Formatted string with relevant tickets
        """
        # Top-k semantic retrieval by embedding similarity.
        results = self.vectorstore.similarity_search(query, k=3)
        
        if not results:
            return "No similar tickets found."
        
        output = "Found similar tickets:\n\n"
        for i, doc in enumerate(results, 1):
            output += f"--- Ticket {i} ---\n"
            output += doc.page_content + "\n\n"
        
        return output
    
    def get_ticket_by_id(self, ticket_id: str) -> str:
        """
        Retrieve a specific ticket by its ID.
        
        Args:
            ticket_id: The ticket ID (e.g., TICK-001)
            
        Returns:
            Formatted ticket details or error message
        """
        # Normalize input to avoid case/whitespace mismatches.
        ticket_id = ticket_id.upper().strip()
        
        for ticket in self.tickets:
            if ticket['ticket_id'] == ticket_id:
                return f"""Ticket ID: {ticket['ticket_id']}
Title: {ticket['title']}
Description: {ticket['description']}
Resolution: {ticket['resolution']}
Category: {ticket['category']}
Priority: {ticket['priority']}
Created: {ticket['created_date']}
Resolved: {ticket['resolved_date']}"""
        
        return f"Ticket {ticket_id} not found in the database."
    
    def search_by_category(self, category: str) -> str:
        """
        Find all tickets in a specific category.
        
        Args:
            category: Category name (e.g., Authentication, Database, Payment)
            
        Returns:
            List of tickets in the category
        """
        # Normalize category string for case-insensitive exact matching.
        category = category.strip()
        matching = [t for t in self.tickets if t['category'].lower() == category.lower()]
        
        if not matching:
            available = list(set(t['category'] for t in self.tickets))
            return f"No tickets found in category '{category}'. Available categories: {', '.join(available)}"
        
        output = f"Found {len(matching)} tickets in category '{category}':\n\n"
        for ticket in matching:
            output += f"• [{ticket['ticket_id']}] {ticket['title']} (Priority: {ticket['priority']})\n"
        
        return output
    
    def get_ticket_statistics(self, input: str = "") -> str:
        """
        Get statistics and insights about the ticket database.
        
        Args:
            input: Optional parameter (not used, but required by Tool interface)
            
        Returns:
            Statistics about tickets
        """
        total = len(self.tickets)
        
        # Aggregate counts for dashboard/overview style questions.
        # Using dict accumulation keeps this dependency-free and explicit.
        categories = {}
        priorities = {}
        for ticket in self.tickets:
            categories[ticket['category']] = categories.get(ticket['category'], 0) + 1
            priorities[ticket['priority']] = priorities.get(ticket['priority'], 0) + 1
        
        output = f"Ticket Database Statistics:\n"
        output += f"Total Tickets: {total}\n\n"
        
        output += "By Category:\n"
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            output += f"  • {cat}: {count}\n"
        
        output += "\nBy Priority:\n"
        for pri, count in sorted(priorities.items(), key=lambda x: x[1], reverse=True):
            output += f"  • {pri}: {count}\n"
        
        return output
    
    def get_tools(self) -> List[Tool]:
        """
        Get all tools as LangChain Tool objects.
        
        Returns:
            List of Tool objects for the agent
        """
        # Tool descriptions are part of the model prompt context.
        # High-quality descriptions materially improve tool-selection accuracy.
        return [
            Tool(
                name="SearchSimilarTickets",
                func=self.search_similar_tickets,
                description="""Use this tool to search for similar support tickets based on a problem description or question.
Input should be a clear description of the issue or question.
This is the PRIMARY tool for answering "how to fix" or "similar issue" questions."""
            ),
            Tool(
                name="GetTicketByID",
                func=self.get_ticket_by_id,
                description="""Use this tool to retrieve details of a specific ticket by its ID.
Input should be a ticket ID like 'TICK-001'.
Use this when the user mentions a specific ticket number."""
            ),
            Tool(
                name="SearchByCategory",
                func=self.search_by_category,
                description="""Use this tool to find all tickets in a specific category.
Input should be a category name like 'Authentication', 'Database', 'Payment', etc.
Use this when the user wants to see all issues of a certain type."""
            ),
            Tool(
                name="GetTicketStatistics",
                func=self.get_ticket_statistics,
                description="""Use this tool to get statistics about the ticket database including total count, 
category distribution, and priority breakdown.
No input needed - just call this tool.
Use this when the user asks for statistics, summaries, or overview of tickets."""
            )
        ]
