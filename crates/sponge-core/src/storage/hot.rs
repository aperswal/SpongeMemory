use dashmap::DashMap;
use moka::sync::Cache;
use std::time::Duration;
use uuid::Uuid;

use super::Storage;
use crate::constants::HOT_IDLE_TIMEOUT_SECS;
use crate::types::MemoryEntry;

/// Hot storage - in-memory with LRU eviction
pub struct HotStorage {
    /// Main storage - DashMap for concurrent access
    entries: DashMap<Uuid, MemoryEntry>,
    /// LRU tracking cache - when entry expires here, it's a candidate for cold storage
    lru: Cache<Uuid, ()>,
    /// User index for fast lookups
    user_index: DashMap<String, Vec<Uuid>>,
}

impl HotStorage {
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: DashMap::new(),
            lru: Cache::builder()
                .max_capacity(capacity as u64)
                .time_to_idle(Duration::from_secs(HOT_IDLE_TIMEOUT_SECS))
                .build(),
            user_index: DashMap::new(),
        }
    }

    /// Touch an entry to mark it as recently used
    pub fn touch(&self, id: &Uuid) {
        self.lru.get(id);
    }

    /// Evict a specific entry from hot storage
    pub fn evict(&self, id: &Uuid) -> Option<MemoryEntry> {
        self.lru.invalidate(id);

        if let Some((_, entry)) = self.entries.remove(id) {
            // Update user index
            if let Some(mut ids) = self.user_index.get_mut(&entry.user_id) {
                ids.retain(|eid| eid != id);
            }
            return Some(entry);
        }

        None
    }
}

impl Storage for HotStorage {
    fn get(&self, id: &Uuid) -> Option<MemoryEntry> {
        // Touch LRU on access
        self.lru.get(id);
        self.entries.get(id).map(|e| e.clone())
    }

    fn put(&self, entry: MemoryEntry) {
        let id = entry.id;
        let user_id = entry.user_id.clone();

        // Add to LRU tracking
        self.lru.insert(id, ());

        // Update user index
        self.user_index
            .entry(user_id)
            .or_default()
            .push(id);

        // Store entry
        self.entries.insert(id, entry);
    }

    fn remove(&self, id: &Uuid) -> Option<MemoryEntry> {
        self.lru.invalidate(id);

        if let Some((_, entry)) = self.entries.remove(id) {
            // Update user index
            if let Some(mut ids) = self.user_index.get_mut(&entry.user_id) {
                ids.retain(|eid| eid != id);
            }
            return Some(entry);
        }

        None
    }

    fn contains(&self, id: &Uuid) -> bool {
        self.entries.contains_key(id)
    }

    fn get_by_user(&self, user_id: &str) -> Vec<MemoryEntry> {
        self.user_index
            .get(user_id)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.entries.get(id).map(|e| e.clone()))
                    .collect()
            })
            .unwrap_or_default()
    }

    fn all_ids(&self) -> Vec<Uuid> {
        self.entries.iter().map(|e| *e.key()).collect()
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn clear(&self) {
        self.entries.clear();
        self.lru.invalidate_all();
        self.user_index.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_entry(user_id: &str, content: &str) -> MemoryEntry {
        MemoryEntry::new(user_id.to_string(), content.to_string(), vec![0.0; 10])
    }

    #[test]
    fn test_put_and_get() {
        let storage = HotStorage::new(100);
        let entry = create_test_entry("user1", "test content");
        let id = entry.id;

        storage.put(entry.clone());

        let retrieved = storage.get(&id);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().content, "test content");
    }

    #[test]
    fn test_get_nonexistent() {
        let storage = HotStorage::new(100);
        let id = Uuid::new_v4();

        assert!(storage.get(&id).is_none());
    }

    #[test]
    fn test_remove() {
        let storage = HotStorage::new(100);
        let entry = create_test_entry("user1", "test content");
        let id = entry.id;

        storage.put(entry);
        assert!(storage.contains(&id));

        let removed = storage.remove(&id);
        assert!(removed.is_some());
        assert!(!storage.contains(&id));
    }

    #[test]
    fn test_remove_nonexistent() {
        let storage = HotStorage::new(100);
        let id = Uuid::new_v4();

        assert!(storage.remove(&id).is_none());
    }

    #[test]
    fn test_get_by_user() {
        let storage = HotStorage::new(100);

        // Add multiple entries for user1
        for i in 0..3 {
            storage.put(create_test_entry("user1", &format!("content {}", i)));
        }

        // Add entries for user2
        for i in 0..2 {
            storage.put(create_test_entry("user2", &format!("other {}", i)));
        }

        let user1_entries = storage.get_by_user("user1");
        assert_eq!(user1_entries.len(), 3);

        let user2_entries = storage.get_by_user("user2");
        assert_eq!(user2_entries.len(), 2);

        let unknown_entries = storage.get_by_user("unknown");
        assert!(unknown_entries.is_empty());
    }

    #[test]
    fn test_len_and_is_empty() {
        let storage = HotStorage::new(100);

        assert!(storage.is_empty());
        assert_eq!(storage.len(), 0);

        storage.put(create_test_entry("user1", "content"));
        assert!(!storage.is_empty());
        assert_eq!(storage.len(), 1);
    }

    #[test]
    fn test_clear() {
        let storage = HotStorage::new(100);

        for i in 0..5 {
            storage.put(create_test_entry("user1", &format!("content {}", i)));
        }

        assert_eq!(storage.len(), 5);

        storage.clear();

        assert!(storage.is_empty());
        assert!(storage.get_by_user("user1").is_empty());
    }

    #[test]
    fn test_all_ids() {
        let storage = HotStorage::new(100);

        let mut inserted_ids = Vec::new();
        for i in 0..5 {
            let entry = create_test_entry("user1", &format!("content {}", i));
            inserted_ids.push(entry.id);
            storage.put(entry);
        }

        let all_ids = storage.all_ids();
        assert_eq!(all_ids.len(), 5);

        for id in inserted_ids {
            assert!(all_ids.contains(&id));
        }
    }

    #[test]
    fn test_contains() {
        let storage = HotStorage::new(100);
        let entry = create_test_entry("user1", "content");
        let id = entry.id;

        assert!(!storage.contains(&id));

        storage.put(entry);

        assert!(storage.contains(&id));
    }

    #[test]
    fn test_user_index_updated_on_remove() {
        let storage = HotStorage::new(100);
        let entry = create_test_entry("user1", "content");
        let id = entry.id;

        storage.put(entry);
        assert_eq!(storage.get_by_user("user1").len(), 1);

        storage.remove(&id);
        assert!(storage.get_by_user("user1").is_empty());
    }

    #[test]
    fn test_evict() {
        let storage = HotStorage::new(100);
        let entry = create_test_entry("user1", "content");
        let id = entry.id;

        storage.put(entry);

        let evicted = storage.evict(&id);
        assert!(evicted.is_some());
        assert!(!storage.contains(&id));
    }
}
