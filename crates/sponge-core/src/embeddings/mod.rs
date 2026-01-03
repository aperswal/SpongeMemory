mod voyage;
mod openai;
mod cohere;
mod google;
mod rate_limited;
mod cached;

pub use voyage::VoyageProvider;
pub use openai::OpenAIProvider;
pub use cohere::CohereProvider;
pub use google::GoogleProvider;
pub use rate_limited::{RateLimitedProvider, RateLimitConfig, ApiStats};
pub use cached::{CachedProvider, CacheStats};

use async_trait::async_trait;
use thiserror::Error;

use crate::config::EmbeddingConfig;

#[derive(Error, Debug)]
pub enum EmbeddingError {
    #[error("API request failed: {0}")]
    RequestFailed(#[from] reqwest::Error),
    #[error("API returned error: {0}")]
    ApiError(String),
    #[error("Invalid response format: {0}")]
    InvalidResponse(String),
    #[error("Rate limited, retry after {0} seconds")]
    RateLimited(u64),
}

/// Trait for embedding providers
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Generate embedding for a single text (default, used for documents)
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError>;

    /// Generate embeddings for multiple texts (batch, used for documents)
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError>;

    /// Get the dimension of embeddings produced by this provider
    fn dimension(&self) -> usize;

    /// Embed text optimized for storage/documents.
    /// Default implementation calls embed(). Override for providers that support
    /// asymmetric embeddings (like Google Gemini with TaskType).
    async fn embed_for_document(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        self.embed(text).await
    }

    /// Embed text optimized for search queries.
    /// Default implementation calls embed(). Override for providers that support
    /// asymmetric embeddings (like Google Gemini with TaskType).
    async fn embed_for_query(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        self.embed(text).await
    }

    /// Batch embed texts optimized for storage/documents.
    async fn embed_batch_for_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        self.embed_batch(texts).await
    }
}

/// Create an embedding provider from config
pub fn create_provider(config: &EmbeddingConfig) -> Box<dyn EmbeddingProvider> {
    match config {
        EmbeddingConfig::Voyage { api_key, model } => {
            Box::new(VoyageProvider::new(api_key.clone(), model.clone()))
        }
        EmbeddingConfig::OpenAI { api_key, model } => {
            Box::new(OpenAIProvider::new(api_key.clone(), model.clone()))
        }
        EmbeddingConfig::Cohere { api_key, model } => {
            Box::new(CohereProvider::new(api_key.clone(), model.clone()))
        }
        EmbeddingConfig::Google { api_key, model } => {
            Box::new(GoogleProvider::new(api_key.clone(), model.clone()))
        }
    }
}

/// Create a rate-limited embedding provider from config
///
/// This wraps the standard provider with rate limiting and exponential backoff
/// to handle API rate limits gracefully.
pub fn create_rate_limited_provider(
    config: &EmbeddingConfig,
    rate_config: Option<RateLimitConfig>,
) -> RateLimitedProvider {
    let inner = create_provider(config);
    RateLimitedProvider::new(inner, rate_config.unwrap_or_default())
}

/// Create a cached embedding provider from config
///
/// Wraps the standard provider with disk-based caching.
/// Useful for tests and development to avoid repeated API calls.
pub fn create_cached_provider(
    config: &EmbeddingConfig,
    cache_dir: impl AsRef<std::path::Path>,
) -> CachedProvider {
    let inner = create_provider(config);
    CachedProvider::new(inner, cache_dir)
}
