"""
Shared pytest fixtures for Shop4You test suite.
"""
import pytest
from vector_store import create_vector_store
from agents import compile_graph


@pytest.fixture(scope="session")
def vs():
    """Load the ChromaDB vector store once per test session."""
    return create_vector_store()


@pytest.fixture(scope="session")
def agent():
    """Compile the LangGraph agent once per test session (in-memory mode)."""
    return compile_graph(use_memory=True, persist=False)
