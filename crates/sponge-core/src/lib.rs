//! # Sponge Core
//!
//! Biologically-inspired memory system for AI applications.
//!
//! ## Features
//!
//! - **HNSW Index**: Fast approximate nearest neighbor search
//! - **Hot/Cold Storage**: In-memory hot tier with disk-based cold storage
//! - **Automatic Forgetting**: Memories decay over time without reinforcement
//! - **Active Recall**: Accessing memories strengthens them
//! - **Background Consolidation**: Periodic cleanup and optimization
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use sponge_core::{Memory, SpongeConfig, EmbeddingConfig};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let config = SpongeConfig {
//!         embedding: EmbeddingConfig::OpenAI {
//!             api_key: std::env::var("OPENAI_API_KEY")?,
//!             model: "text-embedding-3-large".to_string(),
//!         },
//!         ..Default::default()
//!     };
//!
//!     let memory = Memory::new(config).await?;
//!
//!     // Store a memory
//!     memory.remember("user_123", "User prefers dark mode", None).await?;
//!
//!     // Recall memories (active recall - strengthens accessed memories)
//!     let results = memory.recall("user_123", "What are the user's preferences?", 5).await?;
//!
//!     for result in results {
//!         println!("{}: {}", result.similarity, result.entry.content);
//!     }
//!
//!     Ok(())
//! }
//! ```

pub mod clock;
pub mod config;
pub mod consolidation;
pub mod constants;
pub mod embeddings;
pub mod index;
pub mod memory;
pub mod scoring;
pub mod storage;
pub mod types;

// Re-exports for convenience
pub use clock::{Clock, SimulatedClock, SystemClock};
pub use config::{
    ConfigValidationError, ConsolidationConfig, EmbeddingConfig, ScoringConfig, SpongeConfig,
    StorageConfig,
};
pub use memory::{Memory, MemoryError};
pub use scoring::Scorer;
pub use storage::{ColdStorage, HotStorage, Storage};
pub use types::{MemoryEntry, MemoryStats, RecallResult};

// Rate limiting re-exports
pub use embeddings::{
    create_rate_limited_provider, ApiStats, RateLimitConfig, RateLimitedProvider,
};

// Caching re-exports
pub use embeddings::{create_cached_provider, CachedProvider, CacheStats};

// Embedding provider trait and implementations (for custom providers and testing)
pub use embeddings::{EmbeddingProvider, GoogleProvider};
