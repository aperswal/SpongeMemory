# Sponge API Reference

> Complete public API reference for sponge-core.

---

## Memory Struct

The main entry point for all operations.

### Constructors

```rust
/// Create a new Memory instance with system clock
pub async fn new(config: SpongeConfig) -> Result<Self, MemoryError>

/// Create with a custom clock (for testing)
pub async fn with_clock(
    config: SpongeConfig,
    clock: Arc<dyn Clock>,
) -> Result<Self, MemoryError>

/// Create with a custom embedder and clock
pub async fn with_embedder(
    config: SpongeConfig,
    clock: Arc<dyn Clock>,
    embedder: Box<dyn EmbeddingProvider>,
) -> Result<Self, MemoryError>
```

### Core Operations

```rust
/// Store a new memory
/// Returns: UUID of the created memory
pub async fn remember(
    &self,
    user_id: &str,
    content: &str,
    metadata: Option<serde_json::Value>,
) -> Result<Uuid, MemoryError>

/// Store multiple memories efficiently (batch embedding)
/// Returns: UUIDs of the created memories
pub async fn remember_batch(
    &self,
    user_id: &str,
    contents: &[&str],
    metadata: Option<serde_json::Value>,
) -> Result<Vec<Uuid>, MemoryError>

/// Search for memories with active recall (boosts retrieved memories)
/// Returns: Top `limit` results sorted by combined_score
pub async fn recall(
    &self,
    user_id: &str,
    query: &str,
    limit: usize,
) -> Result<Vec<RecallResult>, MemoryError>

/// Search for memories without active recall (pure similarity ranking)
/// Returns: Top `limit` results sorted by combined_score (no score boost)
pub async fn recall_readonly(
    &self,
    user_id: &str,
    query: &str,
    limit: usize,
) -> Result<Vec<RecallResult>, MemoryError>
```

### Direct Access

```rust
/// Get a memory by ID (checks hot then cold storage)
pub fn get(&self, id: &Uuid) -> Option<MemoryEntry>

/// Get a memory with its current decayed score
pub fn get_with_decayed_score(&self, id: &Uuid) -> Option<(MemoryEntry, f64)>
```

### Deletion

```rust
/// Delete a specific memory by ID
/// Returns: true if memory existed and was deleted
pub fn forget(&self, id: &Uuid) -> bool

/// Delete all memories for a user
/// Returns: count of deleted memories
pub fn forget_user(&self, user_id: &str) -> usize
```

### Administration

```rust
/// Get current memory statistics
pub fn stats(&self) -> MemoryStats

/// Manually trigger consolidation (moves cold, deletes expired)
pub async fn consolidate(&self)

/// Graceful shutdown (stops background tasks, waits for completion)
pub async fn shutdown(self)
```

### Internal Accessors (for testing/advanced use)

```rust
pub fn clock(&self) -> &Arc<dyn Clock>
pub fn scorer(&self) -> &Arc<Scorer>
pub fn hot_storage(&self) -> &Arc<HotStorage>
pub fn cold_storage(&self) -> &Arc<ColdStorage>
pub fn index(&self) -> &Arc<VectorIndex>
```

---

## Types

### RecallResult

Returned by `recall()` and `recall_readonly()`:

```rust
pub struct RecallResult {
    pub entry: MemoryEntry,      // The full memory entry
    pub similarity: f32,         // Cosine similarity (0.0-1.0)
    pub combined_score: f64,     // (similarity * 0.7) + (decayed_score * 0.3)
}
```

### MemoryEntry

The core memory record:

```rust
pub struct MemoryEntry {
    pub id: Uuid,                           // Unique identifier
    pub user_id: String,                    // User namespace
    pub content: String,                    // Original text
    pub embedding: Vec<f32>,                // Vector embedding
    pub created_at: DateTime<Utc>,          // Creation timestamp
    pub last_accessed: DateTime<Utc>,       // Last retrieval timestamp
    pub access_count: u64,                  // Number of times recalled
    pub score: f64,                         // Current relevance score
    pub metadata: Option<serde_json::Value>, // Custom JSON data
}
```

### MemoryStats

Returned by `stats()`:

```rust
pub struct MemoryStats {
    pub total_memories: usize,   // Total memories
    pub hot_memories: usize,     // Memories in hot storage
    pub cold_memories: usize,    // Memories in cold storage
    pub total_users: usize,      // Unique user count
}
```

### MemoryError

Error type for Memory operations:

```rust
pub enum MemoryError {
    EmbeddingError(EmbeddingError),
    StorageError(String),
    IndexError(String),
}
```

---

## Configuration

### SpongeConfig

Top-level configuration:

```rust
pub struct SpongeConfig {
    pub embedding: EmbeddingConfig,
    pub storage: StorageConfig,
    pub scoring: ScoringConfig,
    pub consolidation: ConsolidationConfig,
    pub embedding_cache_path: Option<PathBuf>,  // Auto-cache embeddings to disk
}
```

### EmbeddingConfig

Embedding provider selection:

```rust
pub enum EmbeddingConfig {
    Voyage { api_key: String, model: String },
    OpenAI { api_key: String, model: String },
    Cohere { api_key: String, model: String },
    Google { api_key: String, model: String },
}
```

### StorageConfig

```rust
pub struct StorageConfig {
    pub data_path: PathBuf,      // Directory for persistent storage
    pub hot_capacity: usize,     // Max entries in hot storage (default: 10,000)
    pub cold_threshold: f64,     // Score below which entries move to cold (default: 0.3)
}
```

### ScoringConfig

```rust
pub struct ScoringConfig {
    pub decay_half_life_hours: f64,  // Time for score to halve (default: 168.0 = 1 week)
    pub access_boost: f64,           // Score added on recall (default: 0.2)
    pub max_score: f64,              // Maximum score cap (default: 2.0)
    pub similarity_weight: f64,      // Weight for similarity in ranking (default: 0.7)
    pub score_weight: f64,           // Weight for score in ranking (default: 0.3)
}

impl ScoringConfig {
    /// Default production config (168h half-life, 0.2 boost, 2.0 max)
    pub fn default() -> Self

    /// Test config with faster decay (24h half-life, 0.5 boost, 5.0 max)
    pub fn test_config() -> Self
}
```

### ConsolidationConfig

```rust
pub struct ConsolidationConfig {
    pub enabled: bool,            // Run background consolidation (default: true)
    pub interval_seconds: u64,    // Consolidation frequency (default: 3600 = 1 hour)
    pub delete_threshold: f64,    // Score below which memories are deleted (default: 0.1)
}
```

---

## Embedding Providers

### EmbeddingProvider Trait

```rust
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Core embedding methods
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError>;
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError>;
    fn dimension(&self) -> usize;

    /// Asymmetric embedding methods (for providers that support task types)
    /// Default implementations call embed()/embed_batch()
    async fn embed_for_document(&self, text: &str) -> Result<Vec<f32>, EmbeddingError>;
    async fn embed_for_query(&self, text: &str) -> Result<Vec<f32>, EmbeddingError>;
    async fn embed_batch_for_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError>;
}
```

> **Note**: Google Gemini overrides the asymmetric methods to use `RETRIEVAL_DOCUMENT` for storage
> and `RETRIEVAL_QUERY` for search, improving search quality. Other providers use symmetric embeddings.

### Factory Functions

```rust
/// Create provider from config
pub fn create_provider(config: &EmbeddingConfig) -> Box<dyn EmbeddingProvider>

/// Create rate-limited provider
pub fn create_rate_limited_provider(
    config: &EmbeddingConfig,
    rate_config: Option<RateLimitConfig>,
) -> RateLimitedProvider

/// Create cached provider (disk-based cache)
pub fn create_cached_provider(
    config: &EmbeddingConfig,
    cache_dir: impl AsRef<Path>,
) -> CachedProvider
```

### Available Providers

| Provider | Model | Dimensions | Asymmetric |
|----------|-------|------------|------------|
| `GoogleProvider` | gemini-embedding-001 | 3072 (default) | ✅ Yes |
| `OpenAIProvider` | text-embedding-3-large | 3072 | No |
| `VoyageProvider` | voyage-3 | 1024 | No |
| `CohereProvider` | embed-english-v3.0 | 1024 | No |

---

## Clock (for testing)

### Clock Trait

```rust
pub trait Clock: Send + Sync {
    fn now(&self) -> DateTime<Utc>;
}
```

### Implementations

```rust
/// System clock (production use)
pub struct SystemClock;

/// Simulated clock (testing use)
pub struct SimulatedClock {
    current: RwLock<DateTime<Utc>>,
}

impl SimulatedClock {
    pub fn new() -> Self                                    // Starts at current time
    pub fn at(time: DateTime<Utc>) -> Self                  // Starts at specific time
    pub fn advance_hours(&self, hours: i64)                 // Move time forward
    pub fn advance_days(&self, days: i64)                   // Move time forward
    pub fn set(&self, time: DateTime<Utc>)                  // Set to specific time
}
```

---

## Scorer

Direct access to scoring logic:

```rust
pub struct Scorer {
    config: ScoringConfig,
    clock: Arc<dyn Clock>,
}

impl Scorer {
    pub fn new(config: ScoringConfig) -> Self
    pub fn with_clock(config: ScoringConfig, clock: Arc<dyn Clock>) -> Self

    /// Calculate current score after decay
    pub fn calculate_decayed_score(&self, entry: &MemoryEntry) -> f64

    /// Calculate boost on access
    pub fn boost_on_access(&self, current_score: f64) -> f64

    /// Calculate combined ranking score
    pub fn combined_score(&self, similarity: f32, entry: &MemoryEntry) -> f64

    /// Apply decay and boost (active recall)
    pub fn record_access(&self, entry: &mut MemoryEntry)
}
```

---

## Usage Examples

### Basic Usage

```rust
use sponge_core::{Memory, SpongeConfig, EmbeddingConfig};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = SpongeConfig {
        embedding: EmbeddingConfig::Google {
            api_key: std::env::var("GOOGLE_API_KEY")?,
            model: "gemini-embedding-001".to_string(),
        },
        ..Default::default()
    };

    let memory = Memory::new(config).await?;

    // Store a memory
    let id = memory.remember("user_123", "I prefer dark mode", None).await?;

    // Search with active recall (strengthens memories)
    let results = memory.recall("user_123", "What are my preferences?", 5).await?;

    for result in results {
        println!("{:.2}: {}", result.combined_score, result.entry.content);
    }

    memory.shutdown().await;
    Ok(())
}
```

### Testing with Simulated Time

```rust
use sponge_core::{Memory, SpongeConfig, ScoringConfig, SimulatedClock};
use std::sync::Arc;

#[tokio::test]
async fn test_decay() {
    let clock = Arc::new(SimulatedClock::new());

    let config = SpongeConfig {
        scoring: ScoringConfig::test_config(),  // Use test config
        ..Default::default()
    };

    let memory = Memory::with_clock(config, clock.clone()).await.unwrap();

    // Store and advance time
    memory.remember("user", "test content", None).await.unwrap();
    clock.advance_days(7);

    // Memory has decayed, but recall will boost it
    let results = memory.recall("user", "test", 5).await.unwrap();
    assert!(!results.is_empty());
}
```
