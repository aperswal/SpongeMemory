use std::sync::Arc;
use thiserror::Error;
use tokio::task::JoinHandle;
use tracing::{debug, info};
use uuid::Uuid;

use crate::clock::{Clock, SystemClock};
use crate::config::SpongeConfig;
use crate::consolidation::Consolidation;
use crate::embeddings::{create_provider, CachedProvider, EmbeddingError, EmbeddingProvider};
use crate::index::VectorIndex;
use crate::scoring::Scorer;
use crate::storage::{ColdStorage, HotStorage, Storage};
use crate::types::{MemoryEntry, MemoryStats, RecallResult};

#[derive(Error, Debug)]
pub enum MemoryError {
    #[error("Embedding error: {0}")]
    Embedding(#[from] EmbeddingError),
    #[error("Storage error: {0}")]
    Storage(String),
    #[error("Not found: {0}")]
    NotFound(String),
    #[error("Configuration error: {0}")]
    Config(String),
}

/// The main Sponge memory system
pub struct Memory {
    /// Embedding provider
    embedder: Box<dyn EmbeddingProvider>,
    /// HNSW vector index
    index: Arc<VectorIndex>,
    /// Hot (in-memory) storage
    hot: Arc<HotStorage>,
    /// Cold (disk) storage
    cold: Arc<ColdStorage>,
    /// Scoring system
    scorer: Arc<Scorer>,
    /// Clock for time operations
    clock: Arc<dyn Clock>,
    /// Consolidation worker
    consolidation: Arc<Consolidation>,
    /// Consolidation task handle
    consolidation_handle: Option<JoinHandle<()>>,
}

impl Memory {
    /// Create a new Memory instance with system clock (production use)
    pub async fn new(config: SpongeConfig) -> Result<Self, MemoryError> {
        Self::with_clock(config, Arc::new(SystemClock)).await
    }

    /// Create a new Memory instance with a custom clock (testing use)
    pub async fn with_clock(
        config: SpongeConfig,
        clock: Arc<dyn Clock>,
    ) -> Result<Self, MemoryError> {
        let embedder = create_provider(&config.embedding);

        // Auto-wrap with CachedProvider if cache path is configured
        let embedder: Box<dyn EmbeddingProvider> = if let Some(ref cache_path) = config.embedding_cache_path {
            info!("Enabling embedding cache at {:?}", cache_path);
            Box::new(CachedProvider::new(embedder, cache_path))
        } else {
            embedder
        };

        Self::with_embedder(config, clock, embedder).await
    }

    /// Create a new Memory instance with a custom embedder (testing/caching use)
    pub async fn with_embedder(
        config: SpongeConfig,
        clock: Arc<dyn Clock>,
        embedder: Box<dyn EmbeddingProvider>,
    ) -> Result<Self, MemoryError> {
        let dimension = embedder.dimension();

        // Create storage
        std::fs::create_dir_all(&config.storage.data_path)
            .map_err(|e| MemoryError::Storage(e.to_string()))?;

        let hot = Arc::new(HotStorage::new(config.storage.hot_capacity));
        let cold = Arc::new(
            ColdStorage::new(config.storage.data_path.join("cold"))
                .map_err(|e| MemoryError::Storage(e.to_string()))?,
        );

        // Create index
        let index = Arc::new(VectorIndex::new(dimension));

        // Rebuild index from existing data
        let mut index_entries = Vec::new();
        for id in hot.all_ids() {
            if let Some(entry) = hot.get(&id) {
                index_entries.push((id, entry.embedding.clone()));
            }
        }
        for id in cold.all_ids() {
            if let Some(entry) = cold.get(&id) {
                index_entries.push((id, entry.embedding.clone()));
            }
        }
        if !index_entries.is_empty() {
            index.insert_batch(&index_entries);
            info!("Rebuilt index with {} entries", index_entries.len());
        }

        // Create scorer with clock
        let scorer = Arc::new(Scorer::with_clock(config.scoring.clone(), clock.clone()));

        // Create consolidation worker (pass cold_threshold from storage config)
        let consolidation = Arc::new(Consolidation::new(
            config.consolidation.clone(),
            config.storage.cold_threshold,
            hot.clone(),
            cold.clone(),
            index.clone(),
            scorer.clone(),
        ));

        // Start consolidation
        let consolidation_handle = if config.consolidation.enabled {
            Some(consolidation.clone().start())
        } else {
            None
        };

        Ok(Self {
            embedder,
            index,
            hot,
            cold,
            scorer,
            clock,
            consolidation,
            consolidation_handle,
        })
    }

    /// Get the clock used by this memory instance
    pub fn clock(&self) -> &Arc<dyn Clock> {
        &self.clock
    }

    /// Get the scorer used by this memory instance
    pub fn scorer(&self) -> &Arc<Scorer> {
        &self.scorer
    }

    /// Get hot storage (for testing)
    pub fn hot_storage(&self) -> &Arc<HotStorage> {
        &self.hot
    }

    /// Get cold storage (for testing)
    pub fn cold_storage(&self) -> &Arc<ColdStorage> {
        &self.cold
    }

    /// Get the index (for testing)
    pub fn index(&self) -> &Arc<VectorIndex> {
        &self.index
    }

    /// Store a new memory
    pub async fn remember(
        &self,
        user_id: &str,
        content: &str,
        metadata: Option<serde_json::Value>,
    ) -> Result<Uuid, MemoryError> {
        debug!("Remembering content for user {}", user_id);

        // Generate embedding optimized for document storage
        let embedding = self.embedder.embed_for_document(content).await?;

        // Create entry with clock time
        let mut entry =
            MemoryEntry::new_at(user_id.to_string(), content.to_string(), embedding.clone(), self.clock.now());

        if let Some(meta) = metadata {
            entry = entry.with_metadata(meta);
        }

        let id = entry.id;

        // Add to index
        self.index.insert(id, embedding);

        // Store in hot storage
        self.hot.put(entry);

        debug!("Stored memory {} for user {}", id, user_id);
        Ok(id)
    }

    /// Store multiple memories in a single batch (much faster than individual remember calls)
    /// Uses batch embedding API for efficiency
    pub async fn remember_batch(
        &self,
        user_id: &str,
        contents: &[&str],
        metadata: Option<serde_json::Value>,
    ) -> Result<Vec<Uuid>, MemoryError> {
        if contents.is_empty() {
            return Ok(vec![]);
        }

        debug!("Batch remembering {} items for user {}", contents.len(), user_id);

        // Generate embeddings in batch, optimized for document storage
        let embeddings = self.embedder.embed_batch_for_documents(contents).await?;

        let mut ids = Vec::with_capacity(contents.len());
        let now = self.clock.now();

        for (content, embedding) in contents.iter().zip(embeddings.into_iter()) {
            let mut entry =
                MemoryEntry::new_at(user_id.to_string(), content.to_string(), embedding.clone(), now);

            if let Some(ref meta) = metadata {
                entry = entry.with_metadata(meta.clone());
            }

            let id = entry.id;
            ids.push(id);

            // Add to index
            self.index.insert(id, embedding);

            // Store in hot storage
            self.hot.put(entry);
        }

        debug!("Batch stored {} memories for user {}", ids.len(), user_id);
        Ok(ids)
    }

    /// Recall memories similar to a query
    /// This implements active recall - accessing memories strengthens them
    pub async fn recall(
        &self,
        user_id: &str,
        query: &str,
        limit: usize,
    ) -> Result<Vec<RecallResult>, MemoryError> {
        debug!("Recalling memories for user {} with query", user_id);

        // Generate query embedding optimized for search
        let query_embedding = self.embedder.embed_for_query(query).await?;

        // Search index
        let search_results = self.index.search(&query_embedding, limit * 2); // Get more to filter

        let mut results = Vec::with_capacity(limit);

        for search_result in search_results {
            // Try hot storage first, then cold
            let entry = self
                .hot
                .get(&search_result.id)
                .or_else(|| self.cold.get(&search_result.id));

            if let Some(mut entry) = entry {
                // Filter by user
                if entry.user_id != user_id {
                    continue;
                }

                // Active recall: boost score on access
                self.scorer.record_access(&mut entry);

                // Update storage with boosted score
                if self.hot.contains(&search_result.id) {
                    self.hot.put(entry.clone());
                } else {
                    // If accessed from cold, promote to hot
                    self.cold.remove(&search_result.id);
                    self.hot.put(entry.clone());
                }

                let combined_score = self.scorer.combined_score(search_result.similarity, &entry);

                results.push(RecallResult {
                    entry,
                    similarity: search_result.similarity,
                    combined_score,
                });

                if results.len() >= limit {
                    break;
                }
            }
        }

        // Sort by combined score
        results.sort_by(|a, b| b.combined_score.partial_cmp(&a.combined_score).unwrap());

        debug!("Recalled {} memories for user {}", results.len(), user_id);
        Ok(results)
    }

    /// Recall memories without active recall (read-only mode)
    /// This returns results based on similarity and current scores but doesn't boost scores.
    /// Useful for benchmarking against static vector search.
    pub async fn recall_readonly(
        &self,
        user_id: &str,
        query: &str,
        limit: usize,
    ) -> Result<Vec<RecallResult>, MemoryError> {
        debug!("Recalling memories (read-only) for user {} with query", user_id);

        // Generate query embedding optimized for search
        let query_embedding = self.embedder.embed_for_query(query).await?;

        // Search index
        let search_results = self.index.search(&query_embedding, limit * 2);

        let mut results = Vec::with_capacity(limit);

        for search_result in search_results {
            // Try hot storage first, then cold
            let entry = self
                .hot
                .get(&search_result.id)
                .or_else(|| self.cold.get(&search_result.id));

            if let Some(entry) = entry {
                // Filter by user
                if entry.user_id != user_id {
                    continue;
                }

                // NO active recall boost - read-only mode
                let combined_score = self.scorer.combined_score(search_result.similarity, &entry);

                results.push(RecallResult {
                    entry,
                    similarity: search_result.similarity,
                    combined_score,
                });

                if results.len() >= limit {
                    break;
                }
            }
        }

        // Sort by combined score
        results.sort_by(|a, b| b.combined_score.partial_cmp(&a.combined_score).unwrap());

        debug!("Recalled {} memories (read-only) for user {}", results.len(), user_id);
        Ok(results)
    }

    /// Get a specific memory by ID
    pub fn get(&self, id: &Uuid) -> Option<MemoryEntry> {
        self.hot.get(id).or_else(|| self.cold.get(id))
    }

    /// Get a memory and apply current decay (doesn't modify storage)
    pub fn get_with_decayed_score(&self, id: &Uuid) -> Option<(MemoryEntry, f64)> {
        self.get(id).map(|entry| {
            let decayed_score = self.scorer.calculate_decayed_score(&entry);
            (entry, decayed_score)
        })
    }

    /// Delete a specific memory
    pub fn forget(&self, id: &Uuid) -> bool {
        let removed = self.hot.remove(id).is_some() || self.cold.remove(id).is_some();
        if removed {
            self.index.remove(id);
        }
        removed
    }

    /// Delete all memories for a user
    pub fn forget_user(&self, user_id: &str) -> usize {
        let mut count = 0;

        // Remove from hot
        for entry in self.hot.get_by_user(user_id) {
            if self.hot.remove(&entry.id).is_some() {
                self.index.remove(&entry.id);
                count += 1;
            }
        }

        // Remove from cold
        for entry in self.cold.get_by_user(user_id) {
            if self.cold.remove(&entry.id).is_some() {
                self.index.remove(&entry.id);
                count += 1;
            }
        }

        count
    }

    /// Get memory statistics
    pub fn stats(&self) -> MemoryStats {
        let hot_count = self.hot.len();
        let cold_count = self.cold.len();

        // Count unique users (this is O(n) but okay for stats)
        let mut users = std::collections::HashSet::new();
        for id in self.hot.all_ids() {
            if let Some(entry) = self.hot.get(&id) {
                users.insert(entry.user_id.clone());
            }
        }
        for id in self.cold.all_ids() {
            if let Some(entry) = self.cold.get(&id) {
                users.insert(entry.user_id.clone());
            }
        }

        MemoryStats {
            total_memories: hot_count + cold_count,
            hot_memories: hot_count,
            cold_memories: cold_count,
            total_users: users.len(),
        }
    }

    /// Manually trigger consolidation
    pub async fn consolidate(&self) {
        self.consolidation.consolidate().await;
    }

    /// Shutdown the memory system gracefully
    pub async fn shutdown(self) {
        self.consolidation.shutdown();
        if let Some(handle) = self.consolidation_handle {
            let _ = handle.await;
        }
        let _ = self.cold.flush();
    }
}
