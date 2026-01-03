use std::sync::Arc;
use tokio::sync::Notify;
use tokio::time::{interval, Duration};
use tracing::{debug, info, warn};

use crate::config::ConsolidationConfig;
use crate::index::VectorIndex;
use crate::scoring::Scorer;
use crate::storage::{ColdStorage, HotStorage, Storage};

/// Background consolidation worker - the "sleep" process
pub struct Consolidation {
    config: ConsolidationConfig,
    /// Cold storage threshold from StorageConfig - memories below this score move to cold
    cold_threshold: f64,
    hot: Arc<HotStorage>,
    cold: Arc<ColdStorage>,
    index: Arc<VectorIndex>,
    scorer: Arc<Scorer>,
    shutdown: Arc<Notify>,
}

impl Consolidation {
    pub fn new(
        config: ConsolidationConfig,
        cold_threshold: f64,
        hot: Arc<HotStorage>,
        cold: Arc<ColdStorage>,
        index: Arc<VectorIndex>,
        scorer: Arc<Scorer>,
    ) -> Self {
        Self {
            config,
            cold_threshold,
            hot,
            cold,
            index,
            scorer,
            shutdown: Arc::new(Notify::new()),
        }
    }

    /// Start the consolidation background task
    pub fn start(self: Arc<Self>) -> tokio::task::JoinHandle<()> {
        let this = self.clone();
        tokio::spawn(async move {
            this.run().await;
        })
    }

    /// Signal shutdown
    pub fn shutdown(&self) {
        self.shutdown.notify_one();
    }

    /// Main consolidation loop
    async fn run(&self) {
        if !self.config.enabled {
            info!("Consolidation disabled");
            return;
        }

        let mut tick = interval(Duration::from_secs(self.config.interval_seconds));

        loop {
            tokio::select! {
                _ = tick.tick() => {
                    self.consolidate().await;
                }
                _ = self.shutdown.notified() => {
                    info!("Consolidation shutting down");
                    break;
                }
            }
        }
    }

    /// Run one consolidation cycle
    pub async fn consolidate(&self) {
        debug!("Starting consolidation cycle");

        let mut moved_to_cold = 0;
        let mut deleted = 0;

        // 1. Process hot storage - move low-score entries to cold
        let hot_ids = self.hot.all_ids();
        for id in hot_ids {
            if let Some(mut entry) = self.hot.get(&id) {
                // Apply decay
                self.scorer.apply_decay(&mut entry);

                // Check if should delete
                if self.scorer.should_delete(&entry, self.config.delete_threshold) {
                    if self.hot.remove(&id).is_some() {
                        self.index.remove(&id);
                        deleted += 1;
                    }
                    continue;
                }

                // Check if should move to cold (using configured threshold)
                if self.scorer.should_move_to_cold(&entry, self.cold_threshold) {
                    if let Some(entry) = self.hot.remove(&id) {
                        self.cold.put(entry);
                        moved_to_cold += 1;
                    }
                } else {
                    // Update the entry with decayed score
                    self.hot.put(entry);
                }
            }
        }

        // 2. Process cold storage - delete very low-score entries
        let cold_ids = self.cold.all_ids();
        for id in cold_ids {
            if let Some(mut entry) = self.cold.get(&id) {
                self.scorer.apply_decay(&mut entry);

                if self.scorer.should_delete(&entry, self.config.delete_threshold) {
                    if self.cold.remove(&id).is_some() {
                        self.index.remove(&id);
                        deleted += 1;
                    }
                } else {
                    // Update with decayed score
                    self.cold.put(entry);
                }
            }
        }

        // 3. Flush cold storage
        if let Err(e) = self.cold.flush() {
            warn!("Failed to flush cold storage: {}", e);
        }

        info!(
            "Consolidation complete: moved {} to cold, deleted {}",
            moved_to_cold, deleted
        );
    }
}
