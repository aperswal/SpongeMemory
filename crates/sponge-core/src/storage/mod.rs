mod hot;
mod cold;

pub use hot::HotStorage;
pub use cold::ColdStorage;

use uuid::Uuid;

use crate::types::MemoryEntry;

/// Trait for storage backends
/// Re-exported for testing access to storage methods
pub trait Storage: Send + Sync {
    /// Get an entry by ID
    fn get(&self, id: &Uuid) -> Option<MemoryEntry>;

    /// Store an entry
    fn put(&self, entry: MemoryEntry);

    /// Remove an entry
    fn remove(&self, id: &Uuid) -> Option<MemoryEntry>;

    /// Check if entry exists
    fn contains(&self, id: &Uuid) -> bool;

    /// Get all entries for a user
    fn get_by_user(&self, user_id: &str) -> Vec<MemoryEntry>;

    /// Get all entry IDs
    fn all_ids(&self) -> Vec<Uuid>;

    /// Get count
    fn len(&self) -> usize;

    /// Check if empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear all entries
    fn clear(&self);
}
