# Sponge Python Client

Python client for [Sponge](https://github.com/yourusername/sponge) - biologically-inspired memory system for AI applications.

## Installation

```bash
pip install sponge-memory
```

## Quick Start

```python
from sponge_client import Sponge

# Connect to local server
memory = Sponge("http://localhost:8080")

# Store a memory
memory.remember("user_123", "User prefers dark mode")

# Recall memories (active recall - strengthens accessed memories)
results = memory.recall("user_123", "What are user preferences?")

for r in results:
    print(f"{r.similarity:.2f}: {r.content}")
```

## Async Usage

```python
from sponge_client import AsyncSponge

async def main():
    async with AsyncSponge("http://localhost:8080") as memory:
        await memory.remember("user_123", "User likes Python")
        results = await memory.recall("user_123", "programming languages")
        print(results)
```

## API Reference

### `Sponge(url, api_key=None, timeout=30.0)`

Create a new Sponge client.

- `url`: Base URL of the Sponge server
- `api_key`: Optional API key for managed cloud
- `timeout`: Request timeout in seconds

### Methods

- `remember(user_id, content, metadata=None) -> UUID` - Store a memory
- `recall(user_id, query, limit=10) -> list[RecallResult]` - Recall similar memories
- `get(memory_id) -> MemoryEntry | None` - Get a specific memory
- `forget(memory_id) -> bool` - Delete a memory
- `forget_user(user_id) -> int` - Delete all memories for a user
- `stats() -> MemoryStats` - Get system statistics
- `health() -> bool` - Check server health

## License

MIT
