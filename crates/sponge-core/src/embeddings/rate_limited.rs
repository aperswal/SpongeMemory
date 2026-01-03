//! Rate-limited embedding provider wrapper
//!
//! Wraps any EmbeddingProvider with rate limiting and exponential backoff.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use tokio::sync::Mutex;
use tokio::time::sleep;
use tracing::{debug, info, warn};

use super::{EmbeddingError, EmbeddingProvider};

/// Configuration for rate limiting
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Minimum delay between API calls (default: 250ms)
    pub min_delay_ms: u64,
    /// Initial backoff delay on rate limit error (default: 1000ms)
    pub initial_backoff_ms: u64,
    /// Maximum backoff delay (default: 32000ms)
    pub max_backoff_ms: u64,
    /// Maximum number of retries before giving up (default: 5)
    pub max_retries: u32,
    /// Whether to log API call statistics
    pub log_stats: bool,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            min_delay_ms: 250,
            initial_backoff_ms: 1000,
            max_backoff_ms: 32000,
            max_retries: 5,
            log_stats: true,
        }
    }
}

/// Statistics about API usage
#[derive(Debug, Default)]
pub struct ApiStats {
    /// Total number of API calls made
    pub total_calls: AtomicU64,
    /// Number of successful calls
    pub successful_calls: AtomicU64,
    /// Number of rate limit errors encountered
    pub rate_limit_errors: AtomicU64,
    /// Number of retries performed
    pub retries: AtomicU64,
    /// Estimated input tokens (approximate)
    pub estimated_tokens: AtomicU64,
}

impl ApiStats {
    /// Log a summary of API usage
    pub fn log_summary(&self) {
        let total = self.total_calls.load(Ordering::Relaxed);
        let successful = self.successful_calls.load(Ordering::Relaxed);
        let rate_limits = self.rate_limit_errors.load(Ordering::Relaxed);
        let retries = self.retries.load(Ordering::Relaxed);
        let tokens = self.estimated_tokens.load(Ordering::Relaxed);

        info!(
            "API Stats: {} total calls, {} successful, {} rate limits, {} retries, ~{} tokens",
            total, successful, rate_limits, retries, tokens
        );
    }

    /// Estimate cost based on token count (very rough estimate)
    /// Gemini embedding API: ~$0.00001 per 1K tokens
    pub fn estimated_cost(&self) -> f64 {
        let tokens = self.estimated_tokens.load(Ordering::Relaxed) as f64;
        (tokens / 1000.0) * 0.00001
    }
}

/// Rate-limited wrapper for any embedding provider
pub struct RateLimitedProvider {
    inner: Box<dyn EmbeddingProvider>,
    config: RateLimitConfig,
    last_call: Arc<Mutex<Instant>>,
    stats: Arc<ApiStats>,
}

impl RateLimitedProvider {
    /// Create a new rate-limited provider
    pub fn new(inner: Box<dyn EmbeddingProvider>, config: RateLimitConfig) -> Self {
        Self {
            inner,
            config,
            last_call: Arc::new(Mutex::new(Instant::now() - Duration::from_secs(10))),
            stats: Arc::new(ApiStats::default()),
        }
    }

    /// Create with default rate limit config
    pub fn with_defaults(inner: Box<dyn EmbeddingProvider>) -> Self {
        Self::new(inner, RateLimitConfig::default())
    }

    /// Get API usage statistics
    pub fn stats(&self) -> &ApiStats {
        &self.stats
    }

    /// Wait for rate limit delay
    async fn wait_for_rate_limit(&self) {
        let mut last_call = self.last_call.lock().await;
        let elapsed = last_call.elapsed();
        let min_delay = Duration::from_millis(self.config.min_delay_ms);

        if elapsed < min_delay {
            let wait_time = min_delay - elapsed;
            debug!("Rate limiting: waiting {:?}", wait_time);
            sleep(wait_time).await;
        }

        *last_call = Instant::now();
    }

    /// Estimate token count for text (rough approximation: ~4 chars per token)
    fn estimate_tokens(text: &str) -> u64 {
        (text.len() as u64).div_ceil(4)
    }
}

#[async_trait]
impl EmbeddingProvider for RateLimitedProvider {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let tokens = Self::estimate_tokens(text);
        self.stats.estimated_tokens.fetch_add(tokens, Ordering::Relaxed);

        let text = text.to_string();
        let mut backoff_ms = self.config.initial_backoff_ms;
        let mut attempts = 0;

        loop {
            self.wait_for_rate_limit().await;
            self.stats.total_calls.fetch_add(1, Ordering::Relaxed);

            match self.inner.embed(&text).await {
                Ok(result) => {
                    self.stats.successful_calls.fetch_add(1, Ordering::Relaxed);
                    return Ok(result);
                }
                Err(EmbeddingError::RateLimited(retry_after)) => {
                    self.stats.rate_limit_errors.fetch_add(1, Ordering::Relaxed);
                    attempts += 1;

                    if attempts >= self.config.max_retries {
                        warn!(
                            "Rate limit: max retries ({}) exceeded",
                            self.config.max_retries
                        );
                        return Err(EmbeddingError::RateLimited(retry_after));
                    }

                    let wait_ms = (retry_after * 1000).max(backoff_ms);
                    warn!(
                        "Rate limited, retry {} of {}, waiting {}ms",
                        attempts, self.config.max_retries, wait_ms
                    );

                    self.stats.retries.fetch_add(1, Ordering::Relaxed);
                    sleep(Duration::from_millis(wait_ms)).await;
                    backoff_ms = (backoff_ms * 2).min(self.config.max_backoff_ms);
                }
                Err(e) => return Err(e),
            }
        }
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let tokens: u64 = texts.iter().map(|t| Self::estimate_tokens(t)).sum();
        self.stats.estimated_tokens.fetch_add(tokens, Ordering::Relaxed);

        let texts_owned: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
        let mut backoff_ms = self.config.initial_backoff_ms;
        let mut attempts = 0;

        loop {
            self.wait_for_rate_limit().await;
            self.stats.total_calls.fetch_add(1, Ordering::Relaxed);

            let refs: Vec<&str> = texts_owned.iter().map(|s| s.as_str()).collect();
            match self.inner.embed_batch(&refs).await {
                Ok(result) => {
                    self.stats.successful_calls.fetch_add(1, Ordering::Relaxed);
                    return Ok(result);
                }
                Err(EmbeddingError::RateLimited(retry_after)) => {
                    self.stats.rate_limit_errors.fetch_add(1, Ordering::Relaxed);
                    attempts += 1;

                    if attempts >= self.config.max_retries {
                        warn!(
                            "Rate limit: max retries ({}) exceeded",
                            self.config.max_retries
                        );
                        return Err(EmbeddingError::RateLimited(retry_after));
                    }

                    let wait_ms = (retry_after * 1000).max(backoff_ms);
                    warn!(
                        "Rate limited, retry {} of {}, waiting {}ms",
                        attempts, self.config.max_retries, wait_ms
                    );

                    self.stats.retries.fetch_add(1, Ordering::Relaxed);
                    sleep(Duration::from_millis(wait_ms)).await;
                    backoff_ms = (backoff_ms * 2).min(self.config.max_backoff_ms);
                }
                Err(e) => return Err(e),
            }
        }
    }

    fn dimension(&self) -> usize {
        self.inner.dimension()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock provider for testing
    struct MockProvider {
        dimension: usize,
    }

    #[async_trait]
    impl EmbeddingProvider for MockProvider {
        async fn embed(&self, _text: &str) -> Result<Vec<f32>, EmbeddingError> {
            Ok(vec![0.0; self.dimension])
        }

        async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
            Ok(texts.iter().map(|_| vec![0.0; self.dimension]).collect())
        }

        fn dimension(&self) -> usize {
            self.dimension
        }
    }

    #[tokio::test]
    async fn test_rate_limited_provider_basic() {
        let mock = Box::new(MockProvider { dimension: 768 });
        let provider = RateLimitedProvider::with_defaults(mock);

        let result = provider.embed("test text").await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 768);

        let stats = provider.stats();
        assert_eq!(stats.total_calls.load(Ordering::Relaxed), 1);
        assert_eq!(stats.successful_calls.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn test_rate_limited_provider_batch() {
        let mock = Box::new(MockProvider { dimension: 768 });
        let provider = RateLimitedProvider::with_defaults(mock);

        let texts = vec!["text1", "text2", "text3"];
        let result = provider.embed_batch(&texts).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 3);
    }

    #[test]
    fn test_token_estimation() {
        // ~4 chars per token
        assert_eq!(RateLimitedProvider::estimate_tokens("hello"), 2); // 5 chars -> 2 tokens
        assert_eq!(RateLimitedProvider::estimate_tokens("hello world"), 3); // 11 chars -> 3 tokens
        assert_eq!(RateLimitedProvider::estimate_tokens(""), 0);
    }
}
