from typing import Any, Optional
from uuid import UUID

import httpx

from .types import MemoryEntry, MemoryStats, RecallResult


class SpongeError(Exception):
    """Base exception for Sponge client errors."""

    pass


class Sponge:
    """
    Synchronous client for Sponge memory system.

    Example:
        ```python
        memory = Sponge("http://localhost:8080")
        memory.remember("user_123", "User prefers dark mode")
        results = memory.recall("user_123", "preferences")
        ```
    """

    def __init__(
        self,
        url: str = "http://localhost:8080",
        *,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ):
        """
        Initialize Sponge client.

        Args:
            url: Base URL of the Sponge server
            api_key: Optional API key for authentication (for managed cloud)
            timeout: Request timeout in seconds
        """
        self.url = url.rstrip("/")
        self.api_key = api_key
        self._client = httpx.Client(timeout=timeout)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _request(self, method: str, path: str, **kwargs) -> dict:
        url = f"{self.url}{path}"
        response = self._client.request(method, url, headers=self._headers(), **kwargs)

        if response.status_code >= 400:
            try:
                error = response.json().get("error", response.text)
            except Exception:
                error = response.text
            raise SpongeError(f"Request failed: {error}")

        return response.json() if response.text else {}

    def remember(
        self,
        user_id: str,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> UUID:
        """
        Store a new memory.

        Args:
            user_id: User identifier to associate the memory with
            content: The content to remember
            metadata: Optional metadata to attach

        Returns:
            UUID of the created memory
        """
        data = {"user_id": user_id, "content": content}
        if metadata:
            data["metadata"] = metadata

        result = self._request("POST", "/remember", json=data)
        return UUID(result["id"])

    def recall(
        self,
        user_id: str,
        query: str,
        limit: int = 10,
    ) -> list[RecallResult]:
        """
        Recall memories similar to a query.

        This implements active recall - accessed memories are strengthened.

        Args:
            user_id: User identifier to search within
            query: The query to search for
            limit: Maximum number of results

        Returns:
            List of matching memories with similarity scores
        """
        data = {"user_id": user_id, "query": query, "limit": limit}
        result = self._request("POST", "/recall", json=data)
        return [RecallResult(**r) for r in result["results"]]

    def get(self, memory_id: UUID) -> Optional[MemoryEntry]:
        """
        Get a specific memory by ID.

        Args:
            memory_id: UUID of the memory to retrieve

        Returns:
            The memory entry, or None if not found
        """
        try:
            result = self._request("GET", f"/memory/{memory_id}")
            return MemoryEntry(**result)
        except SpongeError:
            return None

    def forget(self, memory_id: UUID) -> bool:
        """
        Delete a specific memory.

        Args:
            memory_id: UUID of the memory to delete

        Returns:
            True if the memory was deleted
        """
        result = self._request("DELETE", f"/memory/{memory_id}")
        return result.get("deleted", False)

    def forget_user(self, user_id: str) -> int:
        """
        Delete all memories for a user.

        Args:
            user_id: User identifier

        Returns:
            Number of memories deleted
        """
        result = self._request("DELETE", f"/user/{user_id}")
        return result.get("deleted_count", 0)

    def stats(self) -> MemoryStats:
        """
        Get memory system statistics.

        Returns:
            Current system statistics
        """
        result = self._request("GET", "/stats")
        return MemoryStats(**result["stats"])

    def health(self) -> bool:
        """
        Check if the server is healthy.

        Returns:
            True if healthy
        """
        try:
            response = self._client.get(f"{self.url}/health")
            return response.status_code == 200
        except Exception:
            return False

    def close(self):
        """Close the client connection."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class AsyncSponge:
    """
    Asynchronous client for Sponge memory system.

    Example:
        ```python
        async with AsyncSponge("http://localhost:8080") as memory:
            await memory.remember("user_123", "User prefers dark mode")
            results = await memory.recall("user_123", "preferences")
        ```
    """

    def __init__(
        self,
        url: str = "http://localhost:8080",
        *,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.url = url.rstrip("/")
        self.api_key = api_key
        self._client = httpx.AsyncClient(timeout=timeout)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _request(self, method: str, path: str, **kwargs) -> dict:
        url = f"{self.url}{path}"
        response = await self._client.request(
            method, url, headers=self._headers(), **kwargs
        )

        if response.status_code >= 400:
            try:
                error = response.json().get("error", response.text)
            except Exception:
                error = response.text
            raise SpongeError(f"Request failed: {error}")

        return response.json() if response.text else {}

    async def remember(
        self,
        user_id: str,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> UUID:
        """Store a new memory."""
        data = {"user_id": user_id, "content": content}
        if metadata:
            data["metadata"] = metadata

        result = await self._request("POST", "/remember", json=data)
        return UUID(result["id"])

    async def recall(
        self,
        user_id: str,
        query: str,
        limit: int = 10,
    ) -> list[RecallResult]:
        """Recall memories similar to a query."""
        data = {"user_id": user_id, "query": query, "limit": limit}
        result = await self._request("POST", "/recall", json=data)
        return [RecallResult(**r) for r in result["results"]]

    async def get(self, memory_id: UUID) -> Optional[MemoryEntry]:
        """Get a specific memory by ID."""
        try:
            result = await self._request("GET", f"/memory/{memory_id}")
            return MemoryEntry(**result)
        except SpongeError:
            return None

    async def forget(self, memory_id: UUID) -> bool:
        """Delete a specific memory."""
        result = await self._request("DELETE", f"/memory/{memory_id}")
        return result.get("deleted", False)

    async def forget_user(self, user_id: str) -> int:
        """Delete all memories for a user."""
        result = await self._request("DELETE", f"/user/{user_id}")
        return result.get("deleted_count", 0)

    async def stats(self) -> MemoryStats:
        """Get memory system statistics."""
        result = await self._request("GET", "/stats")
        return MemoryStats(**result["stats"])

    async def health(self) -> bool:
        """Check if the server is healthy."""
        try:
            response = await self._client.get(f"{self.url}/health")
            return response.status_code == 200
        except Exception:
            return False

    async def close(self):
        """Close the client connection."""
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
