//! Central constants for the Sponge memory system.
//!
//! All magic numbers live here. Update these to change system behavior.

// =============================================================================
// SCORING DEFAULTS
// =============================================================================

/// Production decay half-life in hours (1 week)
pub const DECAY_HALF_LIFE_HOURS: f64 = 168.0;

/// Test decay half-life in hours (1 day) - faster for simulations
pub const DECAY_HALF_LIFE_HOURS_TEST: f64 = 24.0;

/// Production access boost - score added when memory is recalled
pub const ACCESS_BOOST: f64 = 0.2;

/// Test access boost - stronger for visible test effects
pub const ACCESS_BOOST_TEST: f64 = 0.5;

/// Production maximum score cap
pub const MAX_SCORE: f64 = 2.0;

/// Test maximum score cap - higher for repeated access tests
pub const MAX_SCORE_TEST: f64 = 5.0;

/// Weight for similarity in combined score calculation
pub const SIMILARITY_WEIGHT: f64 = 0.7;

/// Weight for decayed score in combined score calculation
pub const SCORE_WEIGHT: f64 = 0.3;

// =============================================================================
// STORAGE DEFAULTS
// =============================================================================

/// Maximum entries in hot storage before LRU eviction
pub const HOT_CAPACITY: usize = 10_000;

/// Score threshold below which entries move from hot to cold storage
pub const COLD_THRESHOLD: f64 = 0.3;

/// Score threshold below which entries are permanently deleted
pub const DELETE_THRESHOLD: f64 = 0.1;

/// Hot storage idle timeout in seconds (1 hour)
pub const HOT_IDLE_TIMEOUT_SECS: u64 = 3600;

// =============================================================================
// CONSOLIDATION DEFAULTS
// =============================================================================

/// How often consolidation runs in seconds (1 hour)
pub const CONSOLIDATION_INTERVAL_SECS: u64 = 3600;

// =============================================================================
// HNSW INDEX PARAMETERS
// =============================================================================

/// HNSW M parameter - connections per node
pub const HNSW_MAX_CONNECTIONS: usize = 24;

/// HNSW ef during construction
pub const HNSW_EF_CONSTRUCTION: usize = 400;

/// HNSW maximum layers
pub const HNSW_MAX_LAYERS: usize = 16;

/// HNSW ef during search
pub const HNSW_EF_SEARCH: usize = 32;

/// HNSW initial capacity estimate
pub const HNSW_INITIAL_CAPACITY: usize = 100_000;

// =============================================================================
// EMBEDDING DIMENSIONS
// =============================================================================

/// Google Gemini embedding dimension (full quality)
pub const GEMINI_DIMENSION: usize = 3072;

/// OpenAI text-embedding-3-large dimension
pub const OPENAI_LARGE_DIMENSION: usize = 3072;

/// OpenAI text-embedding-3-small dimension
pub const OPENAI_SMALL_DIMENSION: usize = 1536;

/// OpenAI ada-002 dimension
pub const OPENAI_ADA_DIMENSION: usize = 1536;

/// Voyage embedding dimension (standard models)
pub const VOYAGE_DIMENSION: usize = 1024;

/// Voyage-lite embedding dimension
pub const VOYAGE_LITE_DIMENSION: usize = 512;

/// Cohere embedding dimension (standard models)
pub const COHERE_DIMENSION: usize = 1024;

/// Cohere light model embedding dimension
pub const COHERE_LIGHT_DIMENSION: usize = 384;
