# Sponge

Biologically-inspired memory system for AI applications.

Sponge implements memory the way brains do: memories decay without reinforcement, strengthen when accessed, and consolidate over time. Built in Rust for maximum performance.

## Features

- **HNSW Index** - Fast approximate nearest neighbor search
- **Hot/Cold Storage** - In-memory hot tier with disk-based cold storage
- **Automatic Forgetting** - Memories decay over time without reinforcement
- **Active Recall** - Accessing memories strengthens them
- **Background Consolidation** - Periodic cleanup and optimization ("sleep")
- **Multiple Embedding Providers** - Voyage AI, OpenAI, Cohere, Google

## Quick Start

### Docker (Recommended)

```bash
# 1. Clone the repo
git clone https://github.com/aperswal/SpongeMemory
cd SpongeMemory

# 2. Configure your API key
cp .env.example .env
# Edit .env and set your GOOGLE_API_KEY (or other provider)

# 3. Run
docker-compose up -d

# 4. Test it
curl http://localhost:8080/health
# "ok"
```

Your data persists in a Docker volume. Stop with `docker-compose down`, restart with `docker-compose up -d`.

### From Source

```bash
# Clone and build
git clone https://github.com/aperswal/SpongeMemory
cd SpongeMemory
cargo build --release

# Set API key and run
export GOOGLE_API_KEY=your-key-here
./target/release/sponge
```

## Usage

### Python

```bash
pip install sponge-memory
```

```python
from sponge_client import Sponge

memory = Sponge("http://localhost:8080")

# Store memories
memory.remember("user_123", "User prefers dark mode")
memory.remember("user_123", "User's favorite language is Python")
memory.remember("user_123", "User is working on a machine learning project")

# Recall memories (active recall - strengthens accessed memories)
results = memory.recall("user_123", "What programming does the user do?")

for r in results:
    print(f"{r.similarity:.2f}: {r.content}")
# 0.89: User's favorite language is Python
# 0.76: User is working on a machine learning project
```

### cURL

```bash
# Store a memory
curl -X POST http://localhost:8080/remember \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_123", "content": "User prefers dark mode"}'

# Recall memories
curl -X POST http://localhost:8080/recall \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_123", "query": "user preferences", "limit": 5}'

# Get stats
curl http://localhost:8080/stats
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/remember` | Store a new memory |
| POST | `/recall` | Recall similar memories |
| GET | `/memory/{id}` | Get a specific memory |
| DELETE | `/memory/{id}` | Delete a memory |
| DELETE | `/user/{user_id}` | Delete all memories for a user |
| GET | `/stats` | Get system statistics |
| POST | `/consolidate` | Manually trigger consolidation |

## Configuration

Sponge can be configured via environment variables or `sponge.toml`:

### Environment Variables

```bash
# Embedding provider (set ONE of these)
VOYAGE_API_KEY=your-key      # Recommended - highest quality
OPENAI_API_KEY=your-key      # Most common
COHERE_API_KEY=your-key
GOOGLE_API_KEY=your-key

# Server
HOST=0.0.0.0
PORT=8080
```

### Config File (sponge.toml)

```toml
host = "0.0.0.0"
port = 8080

[embedding]
provider = "voyage"
api_key = "your-key"
model = "voyage-3"

[storage]
data_path = "./sponge-data"
hot_capacity = 10000
cold_threshold = 0.3

[scoring]
decay_half_life_hours = 168  # 1 week
access_boost = 0.2
max_score = 2.0

[consolidation]
enabled = true
interval_seconds = 3600     # 1 hour
delete_threshold = 0.1
```

## How It Works

### Memory Lifecycle

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   REMEMBER  │────▶│  HOT TIER   │────▶│  COLD TIER  │────▶ FORGOTTEN
└─────────────┘     └─────────────┘     └─────────────┘
                          │                    │
                          │◀───────────────────┘
                          │      (on recall)
                    ACTIVE RECALL
                    (strengthens memory)
```

1. **Remember**: New memories go to hot storage with score 1.0
2. **Decay**: Scores decay exponentially based on half-life
3. **Active Recall**: Accessing a memory boosts its score
4. **Consolidation**: Background "sleep" process:
   - Moves low-score memories to cold storage
   - Deletes very low-score memories
   - Promotes recalled cold memories back to hot

### Scoring

```
score(t) = initial_score × 0.5^(hours_elapsed / half_life) + access_boosts
```

- Memories start with score 1.0
- Each access adds 0.2 (configurable)
- Score decays with configurable half-life (default: 1 week)
- Score capped at 2.0

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Sponge Server                          │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                    REST API (Axum)                      ││
│  └─────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────┐│
│  │                    sponge-core                          ││
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌───────────┐  ││
│  │  │  HNSW   │  │   Hot   │  │  Cold   │  │ Embedding │  ││
│  │  │  Index  │  │ Storage │  │ Storage │  │  Provider │  ││
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └─────┬─────┘  ││
│  │       │            │            │             │        ││
│  │  ┌────┴────────────┴────────────┴─────────────┴─────┐  ││
│  │  │              Scoring + Consolidation              │  ││
│  │  └───────────────────────────────────────────────────┘  ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## Embedding Providers

| Provider | Model | Dimensions | Notes |
|----------|-------|------------|-------|
| Google Gemini | gemini-embedding-001 | 3072 | Free tier (1500 req/min), asymmetric search |
| OpenAI | text-embedding-3-large | 3072 | Most common |
| Voyage AI | voyage-3 | 1024 | Highest quality |
| Cohere | embed-english-v3.0 | 1024 | Good value |

## Deployment

### Docker Compose (Self-Hosted)

```bash
# Configure
cp .env.example .env
# Edit .env with your API key

# Run
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

Data is persisted in a Docker volume (`sponge-data`). Your memories survive restarts.

## Development

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build
cargo build

# Run tests
cargo test

# Run with debug logging
RUST_LOG=debug cargo run
```

## Performance

Target latencies (achievable with proper hardware):

| Operation | Hot Hit | Cold Hit |
|-----------|---------|----------|
| recall() | <5ms | <15ms |
| remember() | <50ms | <50ms |

*Embedding API latency adds 20-100ms depending on provider*

## License

BSL-1.1 (Business Source License)

Free to use and self-host. Contact us for commercial cloud usage.

## Acknowledgments

Inspired by neuroscience research on memory consolidation, the spacing effect, and Hebbian learning.
