# Sponge Database Architecture

> A biologically-inspired memory system that treats retrieval as a write operation.

This document describes the actual database implementation that powers the thesis tests.

---

## Overview

Sponge implements a **two-tier hybrid storage system** that mimics biological memory consolidation. Memories naturally decay over time, but retrieval strengthens them—just like human memory.

```
┌─────────────────────────────────────────────────────────────────┐
│                         MEMORY API                               │
│         remember() / recall() / forget() / consolidate()        │
└─────────────────────────────────────────────────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
        ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
        │   EMBEDDER   │ │    INDEX     │ │   SCORER     │
        │  (Gemini)    │ │   (HNSW)     │ │  (Decay)     │
        └──────────────┘ └──────────────┘ └──────────────┘
                                │
                ┌───────────────┴───────────────┐
                ▼                               ▼
        ┌──────────────┐                ┌──────────────┐
        │  HOT STORAGE │ ◄─── promote ──│ COLD STORAGE │
        │  (In-Memory) │ ─── demote ───►│   (Sled DB)  │
        └──────────────┘                └──────────────┘
```

---

## Storage Tiers

### Tier 1: Hot Storage (In-Memory)

| Property | Value |
|----------|-------|
| **Technology** | DashMap (lock-free concurrent hashmap) + Moka LRU cache |
| **Location** | RAM |
| **Capacity** | 10,000 entries (configurable) |
| **Eviction** | LRU with 1-hour idle timeout |
| **Persistence** | None (lost on shutdown) |

**Purpose**: Fast access to frequently-used memories.

**Data Structures**:
```rust
entries: DashMap<Uuid, MemoryEntry>     // Main storage
user_index: DashMap<String, Vec<Uuid>>  // User → memories mapping
lru_cache: Cache<Uuid, ()>              // Tracks access recency
```

### Tier 2: Cold Storage (Disk-Based)

| Property | Value |
|----------|-------|
| **Technology** | Sled (embedded ACID key-value database) |
| **Location** | Disk (`./sponge-data/cold`) |
| **Capacity** | Unlimited |
| **Persistence** | Full ACID guarantees |
| **Format** | JSON-serialized MemoryEntry |

**Purpose**: Long-term storage for decayed but preserved memories.

**Recovery**: User index rebuilt automatically on startup from disk data.

### Memory Lifecycle

```
Score: 1.0                    Score: 0.5                Score: 0.25
   │                             │                          │
   ▼                             ▼                          ▼
┌─────────┐  7 days decay   ┌─────────┐  7 days decay  ┌─────────┐
│   HOT   │ ───────────────►│   HOT   │───────────────►│   HOT   │
│ score=1 │                 │score=0.5│                │score=0.25│
└─────────┘                 └─────────┘                └─────────┘
                                                            │
                                                    score < 0.3
                                                            ▼
                                                      ┌─────────┐
                                                      │  COLD   │
                                                      │score=0.25│
                                                      └─────────┘
                                                            │
                                                    score < 0.1
                                                            ▼
                                                      ┌─────────┐
                                                      │ DELETED │
                                                      └─────────┘
```

**Thresholds**:
- `cold_threshold`: 0.3 (move from hot to cold)
- `delete_threshold`: 0.1 (permanently delete)

---

## Vector Index (HNSW)

### Technology
**HNSW** (Hierarchical Navigable Small World) - an approximate nearest neighbor algorithm.

| Parameter | Value | Description |
|-----------|-------|-------------|
| `MAX_NB_CONNECTION` | 24 | Neighbor connections per node (M parameter) |
| `EF_CONSTRUCTION` | 400 | Effort during index building |
| `NB_LAYER` | 16 | Maximum hierarchy layers |
| `EF_SEARCH` | 32 | Effort during search |

### Performance
| Operation | Complexity | Notes |
|-----------|------------|-------|
| Insert | O(log n) | True incremental, no rebuild |
| Batch Insert | O(k log n) | Parallel insertion |
| Search | O(log n) | Returns top-k with scores |
| Remove | O(1) | Mapping removal only |

### Distance Metric
**Cosine Similarity**:
```
similarity = dot_product(a, b) / (||a|| × ||b||)
distance = 1 - similarity
```

### ID Mapping
```rust
uuid_to_id: HashMap<Uuid, usize>    // UUID → internal HNSW ID
id_to_uuid: HashMap<usize, Uuid>    // Internal HNSW ID → UUID
next_id: AtomicUsize                // Counter for new IDs
```

---

## Scoring System

### Decay Formula

```
decayed_score = initial_score × 0.5^(hours_elapsed / half_life)
```

| Parameter | Production Default | Test Config | Description |
|-----------|-------------------|-------------|-------------|
| `decay_half_life_hours` | 168.0 (1 week) | 24.0 (1 day) | Time for score to halve |
| `access_boost` | 0.2 | 0.5 | Score added on each retrieval |
| `max_score` | 2.0 | 5.0 | Maximum possible score |

> **Note**: Tests use faster decay (24h) and stronger boosts (0.5) to demonstrate
> effects in shorter simulations. Use `ScoringConfig::test_config()` for tests.

**Decay Examples** (with 24-hour half-life, test config):
| Time Elapsed | Score |
|--------------|-------|
| 0 hours | 1.000 |
| 24 hours (1 half-life) | 0.500 |
| 48 hours (2 half-lives) | 0.250 |
| 72 hours (3 half-lives) | 0.125 |
| 168 hours (7 days) | 0.008 |

### Active Recall Boost

When a memory is retrieved via `recall()`:

```rust
fn record_access(&self, entry: &mut MemoryEntry) {
    // 1. Calculate current decayed score
    let decayed = self.calculate_decayed_score(entry);

    // 2. Add access boost
    let boosted = decayed + self.config.access_boost;  // +0.5

    // 3. Cap at maximum
    entry.score = boosted.min(self.config.max_score);  // max 5.0

    // 4. Reset decay timer
    entry.last_accessed = now();

    // 5. Increment counter
    entry.access_count += 1;
}
```

**Example Recovery**:
```
Day 0: Memory created, score = 1.0
Day 7: Score decayed to 0.008 (nearly forgotten)
Day 7: User recalls memory → score = 0.008 + 0.5 = 0.508
Day 7: Memory is now prominent again!
```

### Combined Score Formula

Used for ranking recall results:

```
combined_score = (similarity × 0.7) + (decayed_score × 0.3)
```

| Component | Weight | Range | Description |
|-----------|--------|-------|-------------|
| Similarity | 0.7 | 0.0 - 1.0 | Cosine similarity from HNSW |
| Decayed Score | 0.3 | 0.0 - 5.0 | Current relevance after decay |

**Example Ranking**:
| Memory | Similarity | Decayed Score | Combined | Rank |
|--------|------------|---------------|----------|------|
| A (boosted) | 0.75 | 2.0 | 0.525 + 0.6 = 1.125 | 1st |
| B (fresh) | 0.80 | 1.0 | 0.560 + 0.3 = 0.860 | 2nd |
| C (old) | 0.85 | 0.1 | 0.595 + 0.03 = 0.625 | 3rd |

Memory A ranks first despite lower similarity because it was recently accessed (boosted score).

---

## Memory Entry Structure

```rust
pub struct MemoryEntry {
    // Identity
    pub id: Uuid,                           // Unique identifier
    pub user_id: String,                    // User namespace

    // Content
    pub content: String,                    // Original text
    pub embedding: Vec<f32>,                // 768D vector (Gemini)

    // Temporal
    pub created_at: DateTime<Utc>,          // When stored
    pub last_accessed: DateTime<Utc>,       // Last retrieval time

    // Scoring
    pub score: f64,                         // Current relevance (pre-decay)
    pub access_count: u64,                  // Number of retrievals

    // Extensions
    pub metadata: Option<serde_json::Value>, // Custom data
}
```

**Size Estimate** (with 3072D Gemini embeddings):
- UUID: 16 bytes
- Embedding: 3072 × 4 = 12,288 bytes
- Content: ~200 bytes average
- Overhead: ~100 bytes
- **Total**: ~12.6 KB per memory

---

## Memory API

### Core Operations

| Method | Description | Active Recall |
|--------|-------------|---------------|
| `remember(user_id, content, metadata)` | Store new memory | N/A |
| `remember_batch(user_id, contents, metadata)` | Store multiple | N/A |
| `recall(user_id, query, limit)` | Search and boost | **Yes** |
| `recall_readonly(user_id, query, limit)` | Search only | No |
| `forget(id)` | Delete specific memory | N/A |
| `forget_user(user_id)` | Delete all user memories | N/A |

### Recall Flow

```
recall(user_id, query, limit=10)
         │
         ▼
┌─────────────────────┐
│  1. Embed query     │  query → 768D vector
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  2. HNSW search     │  find top 20 by similarity (limit × 2)
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  3. Filter by user  │  keep only user's memories
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  4. Calculate scores│  combined = sim×0.7 + decay×0.3
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  5. Active recall   │  boost scores, update timestamps
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  6. Sort & return   │  top 10 by combined_score
└─────────────────────┘
```

### Recall Result

```rust
pub struct RecallResult {
    pub entry: MemoryEntry,      // The memory
    pub similarity: f32,         // Cosine similarity (0.0-1.0)
    pub combined_score: f64,     // Final ranking score
}
```

---

## Consolidation (Background Process)

Runs every hour (configurable) to manage memory lifecycle.

### Process

```
┌─────────────────────────────────────────────────────────────┐
│                    CONSOLIDATION CYCLE                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. HOT STORAGE PASS                                        │
│     ├── Apply decay to all entries                          │
│     ├── Delete entries with score < 0.1                     │
│     ├── Move entries with score < 0.3 to cold               │
│     └── Update remaining entries                            │
│                                                              │
│  2. COLD STORAGE PASS                                       │
│     ├── Apply decay to all entries                          │
│     ├── Delete entries with score < 0.1                     │
│     └── Update remaining entries                            │
│                                                              │
│  3. INDEX CLEANUP                                           │
│     └── Remove deleted entries from HNSW                    │
│                                                              │
│  4. PERSISTENCE                                             │
│     └── Flush cold storage to disk                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Configuration

```rust
pub struct ConsolidationConfig {
    pub enabled: bool,           // Default: true
    pub interval_seconds: u64,   // Default: 3600 (1 hour)
    pub delete_threshold: f64,   // Default: 0.1
}
```

---

## Embedding Provider

### Supported Providers

| Provider | Model | Dimensions | Rate Limit |
|----------|-------|------------|------------|
| **Google Gemini** | gemini-embedding-001 | 3072 (default) | 1500 RPM |
| OpenAI | text-embedding-3-large | 3072 | Varies |
| Voyage | voyage-3 | 1024 | Varies |
| Cohere | embed-english-v3.0 | 1024 | Varies |

### Caching Layer

Embedding cache reduces API calls. Enable via config:

```rust
SpongeConfig {
    embedding_cache_path: Some(PathBuf::from(".sponge-cache")),
    ..Default::default()
}
```

- **Key**: SHA256 hash of content + task type
- **Storage**: Disk-based JSON
- **Benefit**: Eliminates duplicate API calls for same content

### Asymmetric Embeddings (Google Gemini)

Gemini supports different embedding modes for better search quality:

| Operation | Task Type | Optimized For |
|-----------|-----------|---------------|
| `remember()` | RETRIEVAL_DOCUMENT | Storage/indexing |
| `recall()` | RETRIEVAL_QUERY | Search queries |

This is handled automatically—no configuration needed.

---

## Configuration

### Full Configuration Structure

```rust
pub struct SpongeConfig {
    pub embedding: EmbeddingConfig,
    pub storage: StorageConfig,
    pub scoring: ScoringConfig,
    pub consolidation: ConsolidationConfig,
    pub embedding_cache_path: Option<PathBuf>,  // Auto-cache embeddings to disk
}
```

### Test Configuration

For testing, use the provided `ScoringConfig::test_config()`:

```rust
// In tests - use the test config for faster decay and visible effects
let scoring = ScoringConfig::test_config();
// Returns: decay_half_life_hours: 24.0, access_boost: 0.5, max_score: 5.0
```

Full test configuration example:

```rust
SpongeConfig {
    embedding: EmbeddingConfig::Google {
        api_key: "...",
        model: "gemini-embedding-001",
    },
    storage: StorageConfig {
        data_path: "/tmp/test",
        hot_capacity: 10000,
        cold_threshold: 0.5,
    },
    scoring: ScoringConfig::test_config(),  // Fast decay for testing
    consolidation: ConsolidationConfig {
        enabled: false,                     // Manual control in tests
        interval_seconds: 3600,
        delete_threshold: 0.1,
    },
}
```

### Production Configuration (Default)

Production uses `ScoringConfig::default()`:

```rust
ScoringConfig {
    decay_half_life_hours: 168.0,   // 1 week (slower decay)
    access_boost: 0.2,              // Gentler boost
    max_score: 2.0,                 // Lower ceiling
    similarity_weight: 0.7,
    score_weight: 0.3,
}
```

---

## Concurrency Model

### Thread Safety

| Component | Technology | Guarantee |
|-----------|------------|-----------|
| Hot Storage | DashMap | Lock-free reads/writes |
| Cold Storage | Sled | Thread-safe DB |
| Index | RwLock | Concurrent reads, exclusive writes |
| Scorer | Stateless | No locking needed |

### Async Runtime

- **Runtime**: Tokio
- **Background Tasks**: `tokio::spawn` for consolidation
- **Graceful Shutdown**: `tokio::select!` with shutdown signal

---

## File Structure

```
sponge-core/src/
├── lib.rs              # Public API exports
├── memory.rs           # Main Memory struct (383 lines)
├── config.rs           # Configuration structs
├── types.rs            # MemoryEntry, RecallResult
├── clock.rs            # SystemClock, SimulatedClock
├── scoring.rs          # Decay and boosting logic
├── index.rs            # HNSW vector index wrapper
├── consolidation.rs    # Background cleanup task
├── embedding/
│   ├── mod.rs          # EmbeddingProvider trait
│   ├── google.rs       # Gemini implementation
│   └── cached.rs       # Caching wrapper
└── storage/
    ├── mod.rs          # Storage trait
    ├── hot.rs          # In-memory DashMap storage
    └── cold.rs         # Sled disk storage
```

---

## Performance Characteristics

### Latency (from Test 6)

| Operation | P50 | P99 | Notes |
|-----------|-----|-----|-------|
| `recall_readonly()` | 124ms | 139ms | Embedding API dominates |
| `recall()` | 126ms | 206ms | +10ms for score updates |
| `remember()` | ~130ms | ~150ms | Embedding + index insert |

### Throughput

| Metric | Value |
|--------|-------|
| Gemini Rate Limit | 1500 RPM |
| Effective QPS | ~25 queries/second |
| Batch Insert | 100 items/request |

### Memory Usage

| Component | Estimate |
|-----------|----------|
| Per memory entry | ~12.6 KB |
| 10,000 memories | ~126 MB |
| HNSW index overhead | ~20% additional |
| Total for 10K | ~151 MB |

---

## Example: Complete Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ Day 1: User stores "I prefer Python for data science"          │
├─────────────────────────────────────────────────────────────────┤
│ 1. Content → Gemini API (RETRIEVAL_DOCUMENT) → 3072D embedding  │
│ 2. MemoryEntry created: score=1.0, access_count=0              │
│ 3. HNSW index updated with embedding                            │
│ 4. Entry stored in hot storage                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ (7 days pass, no access)
┌─────────────────────────────────────────────────────────────────┐
│ Day 8: Memory has decayed                                       │
├─────────────────────────────────────────────────────────────────┤
│ decayed_score = 1.0 × 0.5^(168/24) = 0.0078                     │
│ Memory is nearly forgotten but still exists                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ (user searches)
┌─────────────────────────────────────────────────────────────────┐
│ Day 8: User queries "What programming languages do I like?"    │
├─────────────────────────────────────────────────────────────────┤
│ 1. Query → Gemini API (RETRIEVAL_QUERY) → 3072D vector          │
│ 2. HNSW returns memory with similarity=0.82                     │
│ 3. Combined score = 0.82×0.7 + 0.0078×0.3 = 0.576              │
│ 4. Active recall: score = 0.0078 + 0.5 = 0.508                 │
│ 5. last_accessed = now, access_count = 1                       │
│ 6. Memory is strong again!                                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ (next search)
┌─────────────────────────────────────────────────────────────────┐
│ Day 9: Same query again                                         │
├─────────────────────────────────────────────────────────────────┤
│ Combined score = 0.82×0.7 + 0.508×0.3 = 0.726                   │
│ Memory now ranks HIGHER than before due to boosted score       │
│ Active recall: score = 0.508 + 0.5 = 1.008                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Relationship to Thesis Tests

| Test | Database Feature Validated |
|------|---------------------------|
| **Test 1** | `record_access()` boost mechanism |
| **Test 2** | Score decay + boost over 7 simulated days |
| **Test 3** | Combined scoring pushes unboosted memories down |
| **Test 4** | `recall()` vs `recall_readonly()` comparison |
| **Test 5** | Rapid score accumulation with repeated access |
| **Test 6** | Latency of `recall()` write-back overhead |

The database architecture directly enables the thesis: retrieval is a write operation that reshapes memory salience through the active recall boost mechanism.
