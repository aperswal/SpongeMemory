"""
Sponge - Biologically-inspired memory system for AI applications.

Quick Start:
    ```python
    from sponge_client import Sponge

    # Connect to local server
    memory = Sponge("http://localhost:8080")

    # Store a memory
    memory.remember("user_123", "User prefers dark mode")

    # Recall memories
    results = memory.recall("user_123", "What are user preferences?")
    for r in results:
        print(f"{r.similarity:.2f}: {r.content}")
    ```
"""

from .client import Sponge, AsyncSponge
from .types import MemoryEntry, RecallResult, MemoryStats

__all__ = [
    "Sponge",
    "AsyncSponge",
    "MemoryEntry",
    "RecallResult",
    "MemoryStats",
]

__version__ = "0.1.0"
