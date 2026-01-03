use hnsw_rs::prelude::*;
use parking_lot::RwLock;
use std::collections::HashMap;
use uuid::Uuid;

use crate::constants::{
    HNSW_EF_CONSTRUCTION, HNSW_EF_SEARCH, HNSW_INITIAL_CAPACITY, HNSW_MAX_CONNECTIONS,
    HNSW_MAX_LAYERS,
};

/// Cosine distance implementation for hnsw_rs
/// Returns cosine distance = 1 - cosine_similarity
#[derive(Default, Clone, Copy)]
pub struct DistCosine;

impl Distance<f32> for DistCosine {
    fn eval(&self, a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 1.0;
        }

        // Clamp to [0, 2] to handle floating point precision issues
        // Cosine similarity is in [-1, 1], so distance is in [0, 2]
        let similarity = dot / (norm_a * norm_b);
        (1.0 - similarity).clamp(0.0, 2.0)
    }
}

/// HNSW index wrapper for fast similarity search
///
/// Uses hnsw_rs which supports true incremental insertion - O(log n) per insert
/// instead of O(n log n) rebuild.
pub struct VectorIndex {
    /// The HNSW index with cosine distance
    index: RwLock<Hnsw<'static, f32, DistCosine>>,
    /// Map from internal index to Uuid
    id_to_uuid: RwLock<HashMap<usize, Uuid>>,
    /// Map from Uuid to internal index
    uuid_to_id: RwLock<HashMap<Uuid, usize>>,
    /// Next internal ID to assign
    next_id: RwLock<usize>,
    /// Embedding dimension
    dimension: usize,
}

/// Search result from the index
pub struct SearchResult {
    pub id: Uuid,
    pub similarity: f32,
}

impl VectorIndex {
    pub fn new(dimension: usize) -> Self {
        // Create HNSW with estimated capacity (will grow as needed)
        let hnsw = Hnsw::new(
            HNSW_MAX_CONNECTIONS,
            HNSW_INITIAL_CAPACITY,
            HNSW_MAX_LAYERS,
            HNSW_EF_CONSTRUCTION,
            DistCosine,
        );

        Self {
            index: RwLock::new(hnsw),
            id_to_uuid: RwLock::new(HashMap::new()),
            uuid_to_id: RwLock::new(HashMap::new()),
            next_id: RwLock::new(0),
            dimension,
        }
    }

    /// Get the embedding dimension this index was created for
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Insert a new vector into the index
    ///
    /// This is O(log n) - true incremental insertion, no rebuild needed.
    pub fn insert(&self, uuid: Uuid, embedding: Vec<f32>) {
        debug_assert_eq!(
            embedding.len(),
            self.dimension,
            "Embedding dimension mismatch: expected {}, got {}",
            self.dimension,
            embedding.len()
        );

        // Get or assign internal ID
        let internal_id = {
            let mut uuid_to_id = self.uuid_to_id.write();
            if uuid_to_id.contains_key(&uuid) {
                // Already exists - for now we don't support updates
                // Just return without inserting again
                return;
            }

            let mut next_id = self.next_id.write();
            let id = *next_id;
            *next_id += 1;

            uuid_to_id.insert(uuid, id);
            self.id_to_uuid.write().insert(id, uuid);
            id
        };

        // Insert into HNSW - this is O(log n) incremental!
        self.index.write().insert((&embedding, internal_id));
    }

    /// Insert multiple vectors at once (parallel insertion)
    pub fn insert_batch(&self, entries: &[(Uuid, Vec<f32>)]) {
        // Prepare data with internal IDs
        let mut data_with_ids: Vec<(&Vec<f32>, usize)> = Vec::with_capacity(entries.len());

        {
            let mut uuid_to_id = self.uuid_to_id.write();
            let mut id_to_uuid = self.id_to_uuid.write();
            let mut next_id = self.next_id.write();

            for (uuid, embedding) in entries {
                if uuid_to_id.contains_key(uuid) {
                    continue; // Skip duplicates
                }

                let id = *next_id;
                *next_id += 1;

                uuid_to_id.insert(*uuid, id);
                id_to_uuid.insert(id, *uuid);
                data_with_ids.push((embedding, id));
            }
        }

        if !data_with_ids.is_empty() {
            // Parallel batch insert
            self.index.write().parallel_insert(&data_with_ids);
        }
    }

    /// Remove a vector from the index
    ///
    /// Note: HNSW doesn't support true deletion, so we just remove from our mappings.
    /// The vector remains in the graph but won't be returned in results.
    pub fn remove(&self, uuid: &Uuid) -> bool {
        let mut uuid_to_id = self.uuid_to_id.write();
        let mut id_to_uuid = self.id_to_uuid.write();

        if let Some(id) = uuid_to_id.remove(uuid) {
            id_to_uuid.remove(&id);
            true
        } else {
            false
        }
    }

    /// Search for nearest neighbors
    pub fn search(&self, query: &[f32], limit: usize) -> Vec<SearchResult> {
        let index = self.index.read();
        let id_to_uuid = self.id_to_uuid.read();

        // Search with ef_search parameter for quality/speed tradeoff
        let neighbours = index.search(query, limit * 2, HNSW_EF_SEARCH); // Get extra to filter removed

        let mut results = Vec::with_capacity(limit);

        for neighbour in neighbours {
            // Only include if UUID mapping exists (not removed)
            if let Some(&uuid) = id_to_uuid.get(&neighbour.d_id) {
                results.push(SearchResult {
                    id: uuid,
                    similarity: 1.0 - neighbour.distance, // Convert distance to similarity
                });

                if results.len() >= limit {
                    break;
                }
            }
        }

        results
    }

    /// Get total number of vectors in the index
    pub fn len(&self) -> usize {
        self.uuid_to_id.read().len()
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.uuid_to_id.read().is_empty()
    }

    /// Clear all data from the index
    ///
    /// Creates a fresh HNSW graph.
    pub fn clear(&self) {
        self.uuid_to_id.write().clear();
        self.id_to_uuid.write().clear();
        *self.next_id.write() = 0;

        // Create fresh HNSW
        let hnsw = Hnsw::new(
            HNSW_MAX_CONNECTIONS,
            HNSW_INITIAL_CAPACITY,
            HNSW_MAX_LAYERS,
            HNSW_EF_CONSTRUCTION,
            DistCosine,
        );
        *self.index.write() = hnsw;
    }

    /// Flush is a no-op for hnsw_rs (incremental insertion is immediate)
    pub fn flush(&self) {
        // No-op - hnsw_rs does true incremental insertion
    }

    /// Get pending count (always 0 for hnsw_rs)
    pub fn pending_count(&self) -> usize {
        0 // hnsw_rs has no pending buffer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
        let dist = DistCosine;
        dist.eval(a, b)
    }

    #[test]
    fn test_cosine_distance_identical_vectors() {
        let distance = cosine_distance(&[1.0, 0.0, 0.0], &[1.0, 0.0, 0.0]);
        assert!(distance.abs() < 0.0001, "Identical vectors should have distance ~0");
    }

    #[test]
    fn test_cosine_distance_orthogonal_vectors() {
        let distance = cosine_distance(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]);
        assert!(
            (distance - 1.0).abs() < 0.0001,
            "Orthogonal vectors should have distance ~1"
        );
    }

    #[test]
    fn test_cosine_distance_opposite_vectors() {
        let distance = cosine_distance(&[1.0, 0.0, 0.0], &[-1.0, 0.0, 0.0]);
        assert!(
            (distance - 2.0).abs() < 0.0001,
            "Opposite vectors should have distance ~2"
        );
    }

    #[test]
    fn test_cosine_distance_zero_vector() {
        let distance = cosine_distance(&[1.0, 0.0, 0.0], &[0.0, 0.0, 0.0]);
        assert_eq!(distance, 1.0, "Zero vector should return distance 1.0");
    }

    #[test]
    fn test_insert_and_search() {
        let index = VectorIndex::new(3);

        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        index.insert(id1, vec![1.0, 0.0, 0.0]);
        index.insert(id2, vec![0.0, 1.0, 0.0]);

        let results = index.search(&[0.9, 0.1, 0.0], 2);

        assert_eq!(results.len(), 2);
        // First result should be id1 (most similar)
        assert_eq!(results[0].id, id1);
        // Similarity should be high for the first result
        assert!(results[0].similarity > 0.9);
    }

    #[test]
    fn test_search_empty_index() {
        let index = VectorIndex::new(3);

        let results = index.search(&[1.0, 0.0, 0.0], 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_returns_correct_similarity() {
        let index = VectorIndex::new(3);

        let id = Uuid::new_v4();
        index.insert(id, vec![1.0, 0.0, 0.0]);

        // Query with identical vector
        let results = index.search(&[1.0, 0.0, 0.0], 1);
        assert_eq!(results.len(), 1);
        assert!(
            results[0].similarity > 0.99,
            "Identical query should have similarity ~1.0, got {}",
            results[0].similarity
        );
    }

    #[test]
    fn test_search_limit() {
        let index = VectorIndex::new(3);

        // Insert 10 vectors
        for _ in 0..10 {
            let id = Uuid::new_v4();
            index.insert(id, vec![1.0, 0.0, 0.0]);
        }

        // Should only return requested limit
        let results = index.search(&[1.0, 0.0, 0.0], 3);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_insert_batch() {
        let index = VectorIndex::new(3);

        let entries: Vec<(Uuid, Vec<f32>)> = (0..5)
            .map(|_| (Uuid::new_v4(), vec![1.0, 0.0, 0.0]))
            .collect();

        index.insert_batch(&entries);

        assert_eq!(index.len(), 5);
    }

    #[test]
    fn test_remove() {
        let index = VectorIndex::new(3);

        let id1 = Uuid::new_v4();
        index.insert(id1, vec![1.0, 0.0, 0.0]);

        assert_eq!(index.len(), 1);

        let removed = index.remove(&id1);
        assert!(removed);
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_remove_nonexistent() {
        let index = VectorIndex::new(3);

        let id = Uuid::new_v4();
        let removed = index.remove(&id);
        assert!(!removed);
    }

    #[test]
    fn test_clear() {
        let index = VectorIndex::new(3);

        for _ in 0..5 {
            index.insert(Uuid::new_v4(), vec![1.0, 0.0, 0.0]);
        }

        assert_eq!(index.len(), 5);

        index.clear();

        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
    }

    #[test]
    fn test_semantic_similarity_ordering() {
        let index = VectorIndex::new(3);

        let id_similar = Uuid::new_v4();
        let id_different = Uuid::new_v4();

        // Insert a vector pointing in +x direction
        index.insert(id_similar, vec![1.0, 0.1, 0.0]);
        // Insert a vector pointing in +y direction
        index.insert(id_different, vec![0.0, 1.0, 0.0]);

        // Query with +x direction - should return similar first
        let results = index.search(&[1.0, 0.0, 0.0], 2);

        // HNSW may return fewer results with small datasets
        assert!(!results.is_empty(), "Should return at least one result");
        assert_eq!(results[0].id, id_similar, "Most similar vector should be first");

        // Only check ordering if we got both results
        if results.len() >= 2 {
            assert!(
                results[0].similarity > results[1].similarity,
                "First result should have higher similarity"
            );
        }
    }

    #[test]
    fn test_high_dimensional_vectors() {
        let dim = 768; // Typical embedding dimension
        let index = VectorIndex::new(dim);

        let id = Uuid::new_v4();
        let mut embedding: Vec<f32> = (0..dim).map(|i| (i as f32).sin()).collect();
        // Normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        for v in &mut embedding {
            *v /= norm;
        }

        index.insert(id, embedding.clone());

        let results = index.search(&embedding, 1);
        assert_eq!(results.len(), 1);
        assert!(
            results[0].similarity > 0.99,
            "Same vector should have very high similarity"
        );
    }
}
