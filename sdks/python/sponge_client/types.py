from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel


class MemoryEntry(BaseModel):
    """A memory stored in Sponge."""

    id: UUID
    user_id: str
    content: str
    created_at: datetime
    last_accessed: datetime
    access_count: int
    score: float
    metadata: Optional[dict[str, Any]] = None


class RecallResult(BaseModel):
    """A result from recalling memories."""

    id: UUID
    content: str
    similarity: float
    combined_score: float
    metadata: Optional[dict[str, Any]] = None


class MemoryStats(BaseModel):
    """Statistics about the memory system."""

    total_memories: int
    hot_memories: int
    cold_memories: int
    total_users: int
