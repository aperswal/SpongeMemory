use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A memory entry stored in the system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    /// Unique identifier
    pub id: Uuid,
    /// User/namespace this memory belongs to
    pub user_id: String,
    /// The actual content stored
    pub content: String,
    /// Pre-computed embedding vector
    pub embedding: Vec<f32>,
    /// When this memory was created
    pub created_at: DateTime<Utc>,
    /// When this memory was last accessed
    pub last_accessed: DateTime<Utc>,
    /// Number of times this memory has been recalled.
    /// NOTE: Tracked for analytics/debugging but NOT used in scoring.
    /// Scoring uses time-based decay + access boost, not access frequency.
    pub access_count: u64,
    /// Current relevance score (decays over time, boosted on access)
    pub score: f64,
    /// Optional metadata
    pub metadata: Option<serde_json::Value>,
}

impl MemoryEntry {
    /// Create a new memory entry with the current system time
    pub fn new(user_id: String, content: String, embedding: Vec<f32>) -> Self {
        Self::new_at(user_id, content, embedding, Utc::now())
    }

    /// Create a new memory entry at a specific time (for testing with simulated clock)
    pub fn new_at(
        user_id: String,
        content: String,
        embedding: Vec<f32>,
        at_time: DateTime<Utc>,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            user_id,
            content,
            embedding,
            created_at: at_time,
            last_accessed: at_time,
            access_count: 0,
            score: 1.0,
            metadata: None,
        }
    }

    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = Some(metadata);
        self
    }
}

/// Result returned from a recall operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallResult {
    pub entry: MemoryEntry,
    /// Cosine similarity to the query
    pub similarity: f32,
    /// Combined score (similarity * relevance)
    pub combined_score: f64,
}

/// Statistics about the memory system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    pub total_memories: usize,
    pub hot_memories: usize,
    pub cold_memories: usize,
    pub total_users: usize,
}
