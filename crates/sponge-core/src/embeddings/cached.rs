//! Cached embedding provider for test corpus setup
//!
//! Caches embeddings to disk based on content hash to avoid redundant API calls
//! during development and testing iterations.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use async_trait::async_trait;
use sha2::{Digest, Sha256};
use tracing::{debug, info};

use super::{EmbeddingError, EmbeddingProvider};

/// Cached embedding provider
///
/// Wraps any EmbeddingProvider and caches results to disk.
/// Cache keys are SHA-256 hashes of the input text content.
pub struct CachedProvider {
    inner: Box<dyn EmbeddingProvider>,
    cache_dir: PathBuf,
    /// In-memory cache loaded from disk
    cache: Mutex<HashMap<String, Vec<f32>>>,
    /// Track cache hits/misses
    hits: Mutex<u64>,
    misses: Mutex<u64>,
}

impl CachedProvider {
    /// Create a new cached provider
    ///
    /// # Arguments
    /// * `inner` - The underlying embedding provider
    /// * `cache_dir` - Directory to store cached embeddings (e.g., ".sponge-cache")
    pub fn new(inner: Box<dyn EmbeddingProvider>, cache_dir: impl AsRef<Path>) -> Self {
        let cache_dir = cache_dir.as_ref().to_path_buf();

        // Create cache directory if it doesn't exist
        if !cache_dir.exists() {
            fs::create_dir_all(&cache_dir).ok();
        }

        // Load existing cache
        let cache = Self::load_cache(&cache_dir).unwrap_or_default();
        let cache_size = cache.len();

        if cache_size > 0 {
            info!("Loaded {} cached embeddings from {:?}", cache_size, cache_dir);
        }

        Self {
            inner,
            cache_dir,
            cache: Mutex::new(cache),
            hits: Mutex::new(0),
            misses: Mutex::new(0),
        }
    }

    /// Compute SHA-256 hash of text content with optional task suffix
    fn hash_content(text: &str) -> String {
        Self::hash_content_with_task(text, None)
    }

    /// Compute SHA-256 hash of text content with task type suffix for cache differentiation
    fn hash_content_with_task(text: &str, task: Option<&str>) -> String {
        let mut hasher = Sha256::new();
        hasher.update(text.as_bytes());
        if let Some(task_type) = task {
            hasher.update(b"::");
            hasher.update(task_type.as_bytes());
        }
        let result = hasher.finalize();
        hex::encode(result)
    }

    /// Get cache file path
    fn cache_file_path(&self) -> PathBuf {
        self.cache_dir.join("embeddings.json")
    }

    /// Load cache from disk
    fn load_cache(cache_dir: &Path) -> Option<HashMap<String, Vec<f32>>> {
        let cache_file = cache_dir.join("embeddings.json");
        if !cache_file.exists() {
            return None;
        }

        let file = File::open(&cache_file).ok()?;
        let reader = BufReader::new(file);
        serde_json::from_reader(reader).ok()
    }

    /// Save cache to disk
    fn save_cache(&self) -> Result<(), std::io::Error> {
        let cache = self.cache.lock().unwrap();
        let cache_file = self.cache_file_path();

        let file = File::create(&cache_file)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, &*cache)?;

        debug!("Saved {} embeddings to cache", cache.len());
        Ok(())
    }

    /// Get from cache
    fn get_cached(&self, text: &str) -> Option<Vec<f32>> {
        self.get_cached_with_task(text, None)
    }

    /// Get from cache with task type
    fn get_cached_with_task(&self, text: &str, task: Option<&str>) -> Option<Vec<f32>> {
        let hash = Self::hash_content_with_task(text, task);
        let cache = self.cache.lock().unwrap();
        cache.get(&hash).cloned()
    }

    /// Store in cache
    fn store_cached(&self, text: &str, embedding: Vec<f32>) {
        self.store_cached_with_task(text, None, embedding)
    }

    /// Store in cache with task type
    fn store_cached_with_task(&self, text: &str, task: Option<&str>, embedding: Vec<f32>) {
        let hash = Self::hash_content_with_task(text, task);
        let mut cache = self.cache.lock().unwrap();
        cache.insert(hash, embedding);
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let hits = *self.hits.lock().unwrap();
        let misses = *self.misses.lock().unwrap();
        let cache_size = self.cache.lock().unwrap().len();

        CacheStats {
            hits,
            misses,
            cache_size,
        }
    }

    /// Log cache statistics
    pub fn log_stats(&self) {
        let stats = self.stats();
        let hit_rate = if stats.hits + stats.misses > 0 {
            (stats.hits as f64 / (stats.hits + stats.misses) as f64) * 100.0
        } else {
            0.0
        };

        info!(
            "Embedding cache: {} entries, {} hits, {} misses ({:.1}% hit rate)",
            stats.cache_size, stats.hits, stats.misses, hit_rate
        );
    }

    /// Clear the cache (both in-memory and on disk)
    pub fn clear(&self) -> Result<(), std::io::Error> {
        let mut cache = self.cache.lock().unwrap();
        cache.clear();

        let cache_file = self.cache_file_path();
        if cache_file.exists() {
            fs::remove_file(&cache_file)?;
        }

        info!("Cleared embedding cache");
        Ok(())
    }

    /// Flush in-memory cache to disk
    pub fn flush(&self) -> Result<(), std::io::Error> {
        self.save_cache()
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Number of cache hits
    pub hits: u64,
    /// Number of cache misses
    pub misses: u64,
    /// Current cache size (number of entries)
    pub cache_size: usize,
}

#[async_trait]
impl EmbeddingProvider for CachedProvider {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        // Check cache first
        if let Some(embedding) = self.get_cached(text) {
            *self.hits.lock().unwrap() += 1;
            debug!("Cache hit for text hash: {}", &Self::hash_content(text)[..8]);
            return Ok(embedding);
        }

        // Cache miss - call the underlying provider
        *self.misses.lock().unwrap() += 1;
        debug!("Cache miss for text hash: {}", &Self::hash_content(text)[..8]);

        let embedding = self.inner.embed(text).await?;

        // Store in cache
        self.store_cached(text, embedding.clone());

        // Periodically save cache to disk (every 10 new embeddings)
        let misses = *self.misses.lock().unwrap();
        if misses.is_multiple_of(10) {
            if let Err(e) = self.save_cache() {
                tracing::warn!("Failed to save embedding cache: {}", e);
            }
        }

        Ok(embedding)
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let mut results = Vec::with_capacity(texts.len());
        let mut uncached_texts: Vec<(usize, &str)> = Vec::new();

        // Check cache for each text
        for (i, text) in texts.iter().enumerate() {
            if let Some(embedding) = self.get_cached(text) {
                *self.hits.lock().unwrap() += 1;
                results.push((i, embedding));
            } else {
                uncached_texts.push((i, *text));
            }
        }

        // Fetch uncached embeddings
        if !uncached_texts.is_empty() {
            *self.misses.lock().unwrap() += uncached_texts.len() as u64;

            let uncached_strs: Vec<&str> = uncached_texts.iter().map(|(_, t)| *t).collect();
            let embeddings = self.inner.embed_batch(&uncached_strs).await?;

            // Store in cache and collect results
            for ((i, text), embedding) in uncached_texts.into_iter().zip(embeddings) {
                self.store_cached(text, embedding.clone());
                results.push((i, embedding));
            }

            // Save cache after batch
            if let Err(e) = self.save_cache() {
                tracing::warn!("Failed to save embedding cache: {}", e);
            }
        }

        // Sort by original index and extract embeddings
        results.sort_by_key(|(i, _)| *i);
        Ok(results.into_iter().map(|(_, e)| e).collect())
    }

    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    async fn embed_for_document(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        // Check cache with "document" task type
        if let Some(embedding) = self.get_cached_with_task(text, Some("document")) {
            *self.hits.lock().unwrap() += 1;
            return Ok(embedding);
        }

        *self.misses.lock().unwrap() += 1;
        let embedding = self.inner.embed_for_document(text).await?;
        self.store_cached_with_task(text, Some("document"), embedding.clone());

        let misses = *self.misses.lock().unwrap();
        if misses.is_multiple_of(10) {
            if let Err(e) = self.save_cache() {
                tracing::warn!("Failed to save embedding cache: {}", e);
            }
        }

        Ok(embedding)
    }

    async fn embed_for_query(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        // Check cache with "query" task type
        if let Some(embedding) = self.get_cached_with_task(text, Some("query")) {
            *self.hits.lock().unwrap() += 1;
            return Ok(embedding);
        }

        *self.misses.lock().unwrap() += 1;
        let embedding = self.inner.embed_for_query(text).await?;
        self.store_cached_with_task(text, Some("query"), embedding.clone());

        let misses = *self.misses.lock().unwrap();
        if misses.is_multiple_of(10) {
            if let Err(e) = self.save_cache() {
                tracing::warn!("Failed to save embedding cache: {}", e);
            }
        }

        Ok(embedding)
    }

    async fn embed_batch_for_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let mut results = Vec::with_capacity(texts.len());
        let mut uncached_texts: Vec<(usize, &str)> = Vec::new();

        // Check cache for each text with "document" task type
        for (i, text) in texts.iter().enumerate() {
            if let Some(embedding) = self.get_cached_with_task(text, Some("document")) {
                *self.hits.lock().unwrap() += 1;
                results.push((i, embedding));
            } else {
                uncached_texts.push((i, *text));
            }
        }

        // Fetch uncached embeddings
        if !uncached_texts.is_empty() {
            *self.misses.lock().unwrap() += uncached_texts.len() as u64;

            let uncached_strs: Vec<&str> = uncached_texts.iter().map(|(_, t)| *t).collect();
            let embeddings = self.inner.embed_batch_for_documents(&uncached_strs).await?;

            for ((i, text), embedding) in uncached_texts.into_iter().zip(embeddings) {
                self.store_cached_with_task(text, Some("document"), embedding.clone());
                results.push((i, embedding));
            }

            if let Err(e) = self.save_cache() {
                tracing::warn!("Failed to save embedding cache: {}", e);
            }
        }

        // Sort by original index and extract embeddings
        results.sort_by_key(|(i, _)| *i);
        Ok(results.into_iter().map(|(_, e)| e).collect())
    }
}

impl Drop for CachedProvider {
    fn drop(&mut self) {
        // Save cache on drop
        if let Err(e) = self.save_cache() {
            tracing::warn!("Failed to save embedding cache on drop: {}", e);
        }
        self.log_stats();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock provider for testing
    struct MockProvider {
        dimension: usize,
        call_count: Mutex<u64>,
    }

    impl MockProvider {
        fn new(dimension: usize) -> Self {
            Self {
                dimension,
                call_count: Mutex::new(0),
            }
        }
    }

    #[async_trait]
    impl EmbeddingProvider for MockProvider {
        async fn embed(&self, _text: &str) -> Result<Vec<f32>, EmbeddingError> {
            *self.call_count.lock().unwrap() += 1;
            Ok(vec![0.1; self.dimension])
        }

        async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
            *self.call_count.lock().unwrap() += 1;
            Ok(texts.iter().map(|_| vec![0.1; self.dimension]).collect())
        }

        fn dimension(&self) -> usize {
            self.dimension
        }
    }

    #[tokio::test]
    async fn test_cached_provider_basic() {
        let temp_dir = tempfile::tempdir().unwrap();
        let mock = Box::new(MockProvider::new(768));
        let provider = CachedProvider::new(mock, temp_dir.path());

        // First call - cache miss
        let result1 = provider.embed("test text").await;
        assert!(result1.is_ok());

        // Second call - cache hit
        let result2 = provider.embed("test text").await;
        assert!(result2.is_ok());

        let stats = provider.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.cache_size, 1);
    }

    #[tokio::test]
    async fn test_cached_provider_different_texts() {
        let temp_dir = tempfile::tempdir().unwrap();
        let mock = Box::new(MockProvider::new(768));
        let provider = CachedProvider::new(mock, temp_dir.path());

        let _ = provider.embed("text 1").await;
        let _ = provider.embed("text 2").await;
        let _ = provider.embed("text 1").await; // Hit

        let stats = provider.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 2);
        assert_eq!(stats.cache_size, 2);
    }

    #[test]
    fn test_content_hash() {
        let hash1 = CachedProvider::hash_content("hello");
        let hash2 = CachedProvider::hash_content("hello");
        let hash3 = CachedProvider::hash_content("world");

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
        assert_eq!(hash1.len(), 64); // SHA-256 produces 64 hex chars
    }
}
