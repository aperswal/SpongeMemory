use parking_lot::RwLock;
use sled::Db;
use std::collections::HashMap;
use std::path::Path;
use uuid::Uuid;

use super::Storage;
use crate::types::MemoryEntry;

/// Cold storage - disk-based using sled
pub struct ColdStorage {
    db: Db,
    /// User index stored separately
    user_index: RwLock<HashMap<String, Vec<Uuid>>>,
}

impl ColdStorage {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, sled::Error> {
        let db = sled::open(path)?;

        // Rebuild user index from existing data
        let mut user_index: HashMap<String, Vec<Uuid>> = HashMap::new();
        for (_key, value) in db.iter().flatten() {
            if let Ok(entry) = serde_json::from_slice::<MemoryEntry>(&value) {
                user_index
                    .entry(entry.user_id.clone())
                    .or_default()
                    .push(entry.id);
            }
        }

        Ok(Self {
            db,
            user_index: RwLock::new(user_index),
        })
    }

    /// Flush to disk
    pub fn flush(&self) -> Result<(), sled::Error> {
        self.db.flush()?;
        Ok(())
    }
}

impl Storage for ColdStorage {
    fn get(&self, id: &Uuid) -> Option<MemoryEntry> {
        let key = id.as_bytes();
        self.db
            .get(key)
            .ok()
            .flatten()
            .and_then(|bytes| serde_json::from_slice(&bytes).ok())
    }

    fn put(&self, entry: MemoryEntry) {
        let key = entry.id.as_bytes();
        if let Ok(value) = serde_json::to_vec(&entry) {
            let _ = self.db.insert(key, value);

            // Update user index
            self.user_index
                .write()
                .entry(entry.user_id.clone())
                .or_default()
                .push(entry.id);
        }
    }

    fn remove(&self, id: &Uuid) -> Option<MemoryEntry> {
        let key = id.as_bytes();

        if let Ok(Some(bytes)) = self.db.remove(key) {
            if let Ok(entry) = serde_json::from_slice::<MemoryEntry>(&bytes) {
                // Update user index
                if let Some(ids) = self.user_index.write().get_mut(&entry.user_id) {
                    ids.retain(|eid| eid != id);
                }
                return Some(entry);
            }
        }

        None
    }

    fn contains(&self, id: &Uuid) -> bool {
        let key = id.as_bytes();
        self.db.contains_key(key).unwrap_or(false)
    }

    fn get_by_user(&self, user_id: &str) -> Vec<MemoryEntry> {
        self.user_index
            .read()
            .get(user_id)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }

    fn all_ids(&self) -> Vec<Uuid> {
        self.db
            .iter()
            .filter_map(|result| {
                result.ok().and_then(|(key, _)| {
                    Uuid::from_slice(&key).ok()
                })
            })
            .collect()
    }

    fn len(&self) -> usize {
        self.db.len()
    }

    fn clear(&self) {
        let _ = self.db.clear();
        self.user_index.write().clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn create_test_entry(user_id: &str, content: &str) -> MemoryEntry {
        MemoryEntry::new(user_id.to_string(), content.to_string(), vec![0.0; 10])
    }

    #[test]
    fn test_put_and_get() {
        let dir = tempdir().unwrap();
        let storage = ColdStorage::new(dir.path().join("cold")).unwrap();

        let entry = create_test_entry("user1", "test content");
        let id = entry.id;

        storage.put(entry.clone());

        let retrieved = storage.get(&id);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().content, "test content");
    }

    #[test]
    fn test_get_nonexistent() {
        let dir = tempdir().unwrap();
        let storage = ColdStorage::new(dir.path().join("cold")).unwrap();

        let id = Uuid::new_v4();
        assert!(storage.get(&id).is_none());
    }

    #[test]
    fn test_remove() {
        let dir = tempdir().unwrap();
        let storage = ColdStorage::new(dir.path().join("cold")).unwrap();

        let entry = create_test_entry("user1", "test content");
        let id = entry.id;

        storage.put(entry);
        assert!(storage.contains(&id));

        let removed = storage.remove(&id);
        assert!(removed.is_some());
        assert!(!storage.contains(&id));
    }

    #[test]
    fn test_persistence() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("cold");

        let entry_id;

        // Write and close
        {
            let storage = ColdStorage::new(&path).unwrap();
            let entry = create_test_entry("user1", "persisted content");
            entry_id = entry.id;
            storage.put(entry);
            storage.flush().unwrap();
        }

        // Reopen and read
        {
            let storage = ColdStorage::new(&path).unwrap();
            let retrieved = storage.get(&entry_id);
            assert!(retrieved.is_some());
            assert_eq!(retrieved.unwrap().content, "persisted content");
        }
    }

    #[test]
    fn test_user_index_rebuilt_on_open() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("cold");

        // Write entries
        {
            let storage = ColdStorage::new(&path).unwrap();
            for i in 0..3 {
                storage.put(create_test_entry("user1", &format!("content {}", i)));
            }
            storage.flush().unwrap();
        }

        // Reopen and check user index
        {
            let storage = ColdStorage::new(&path).unwrap();
            let user_entries = storage.get_by_user("user1");
            assert_eq!(user_entries.len(), 3);
        }
    }

    #[test]
    fn test_get_by_user() {
        let dir = tempdir().unwrap();
        let storage = ColdStorage::new(dir.path().join("cold")).unwrap();

        for i in 0..3 {
            storage.put(create_test_entry("user1", &format!("content {}", i)));
        }
        for i in 0..2 {
            storage.put(create_test_entry("user2", &format!("other {}", i)));
        }

        let user1_entries = storage.get_by_user("user1");
        assert_eq!(user1_entries.len(), 3);

        let user2_entries = storage.get_by_user("user2");
        assert_eq!(user2_entries.len(), 2);
    }

    #[test]
    fn test_clear() {
        let dir = tempdir().unwrap();
        let storage = ColdStorage::new(dir.path().join("cold")).unwrap();

        for i in 0..5 {
            storage.put(create_test_entry("user1", &format!("content {}", i)));
        }

        assert_eq!(storage.len(), 5);

        storage.clear();

        assert_eq!(storage.len(), 0);
        assert!(storage.get_by_user("user1").is_empty());
    }

    #[test]
    fn test_all_ids() {
        let dir = tempdir().unwrap();
        let storage = ColdStorage::new(dir.path().join("cold")).unwrap();

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
}
