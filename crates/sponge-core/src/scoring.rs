use std::sync::Arc;

use crate::clock::{Clock, SystemClock};
use crate::config::ScoringConfig;
use crate::types::MemoryEntry;

/// Handles decay and active recall scoring
///
/// The Scorer uses an injectable clock to enable testing time-dependent
/// behavior without waiting real time.
pub struct Scorer {
    config: ScoringConfig,
    clock: Arc<dyn Clock>,
}

impl Scorer {
    /// Create a new Scorer with the system clock (production use)
    pub fn new(config: ScoringConfig) -> Self {
        Self {
            config,
            clock: Arc::new(SystemClock),
        }
    }

    /// Create a new Scorer with a custom clock (testing use)
    pub fn with_clock(config: ScoringConfig, clock: Arc<dyn Clock>) -> Self {
        Self { config, clock }
    }

    /// Get the current time from the clock
    pub fn now(&self) -> chrono::DateTime<chrono::Utc> {
        self.clock.now()
    }

    /// Get the scoring config
    pub fn config(&self) -> &ScoringConfig {
        &self.config
    }

    /// Calculate the current score of a memory entry after decay
    /// Uses exponential decay: score = initial_score * 0.5^(hours_elapsed / half_life)
    pub fn calculate_decayed_score(&self, entry: &MemoryEntry) -> f64 {
        self.calculate_decayed_score_at(entry, self.clock.now())
    }

    /// Calculate decayed score at a specific time (for testing)
    pub fn calculate_decayed_score_at(
        &self,
        entry: &MemoryEntry,
        at_time: chrono::DateTime<chrono::Utc>,
    ) -> f64 {
        let hours_elapsed = (at_time - entry.last_accessed).num_seconds() as f64 / 3600.0;

        if hours_elapsed <= 0.0 {
            return entry.score;
        }

        let decay_factor = 0.5_f64.powf(hours_elapsed / self.config.decay_half_life_hours);
        entry.score * decay_factor
    }

    /// Boost score when a memory is accessed (active recall)
    /// Returns the new score (capped at max_score)
    pub fn boost_on_access(&self, current_score: f64) -> f64 {
        let new_score = current_score + self.config.access_boost;
        new_score.min(self.config.max_score)
    }

    /// Calculate combined score for ranking results
    /// Combines cosine similarity with relevance score using configured weights
    pub fn combined_score(&self, similarity: f32, entry: &MemoryEntry) -> f64 {
        let decayed_score = self.calculate_decayed_score(entry);
        // Use configured weights (default: 0.7 similarity, 0.3 score)
        (similarity as f64 * self.config.similarity_weight) + (decayed_score * self.config.score_weight)
    }

    /// Check if a memory should be moved to cold storage
    pub fn should_move_to_cold(&self, entry: &MemoryEntry, threshold: f64) -> bool {
        self.calculate_decayed_score(entry) < threshold
    }

    /// Check if a memory should be deleted entirely
    pub fn should_delete(&self, entry: &MemoryEntry, threshold: f64) -> bool {
        self.calculate_decayed_score(entry) < threshold
    }

    /// Apply decay to an entry and update its stored score
    /// Also updates last_accessed to prevent double-decay in subsequent calculations
    pub fn apply_decay(&self, entry: &mut MemoryEntry) {
        entry.score = self.calculate_decayed_score(entry);
        entry.last_accessed = self.clock.now();
    }

    /// Record an access (active recall) on an entry
    /// First applies decay to get current effective score, then boosts
    pub fn record_access(&self, entry: &mut MemoryEntry) {
        // First calculate the decayed score based on time since last access
        let decayed_score = self.calculate_decayed_score(entry);

        // Now boost from the decayed score
        entry.access_count += 1;
        entry.last_accessed = self.clock.now();
        entry.score = self.boost_on_access(decayed_score);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::SimulatedClock;
    use crate::constants::{
        ACCESS_BOOST, DECAY_HALF_LIFE_HOURS_TEST, MAX_SCORE, SCORE_WEIGHT, SIMILARITY_WEIGHT,
    };

    fn test_config() -> ScoringConfig {
        ScoringConfig {
            decay_half_life_hours: DECAY_HALF_LIFE_HOURS_TEST,
            access_boost: ACCESS_BOOST,
            max_score: MAX_SCORE,
            similarity_weight: SIMILARITY_WEIGHT,
            score_weight: SCORE_WEIGHT,
        }
    }

    fn create_entry_at_time(clock: &SimulatedClock) -> MemoryEntry {
        let mut entry = MemoryEntry::new(
            "user".to_string(),
            "test content".to_string(),
            vec![0.0; 10],
        );
        entry.score = 1.0;
        entry.last_accessed = clock.now();
        entry.created_at = clock.now();
        entry
    }

    #[test]
    fn test_decay_calculation_fresh() {
        let clock = Arc::new(SimulatedClock::new());
        let scorer = Scorer::with_clock(test_config(), clock.clone());
        let entry = create_entry_at_time(&clock);

        // Score should be 1.0 when just created
        let score = scorer.calculate_decayed_score(&entry);
        assert!(
            (score - 1.0).abs() < 0.001,
            "Fresh entry should have score ~1.0, got {}",
            score
        );
    }

    #[test]
    fn test_decay_calculation_one_half_life() {
        let clock = Arc::new(SimulatedClock::new());
        let scorer = Scorer::with_clock(test_config(), clock.clone());
        let entry = create_entry_at_time(&clock);

        // Advance clock by one half-life (24 hours)
        clock.advance_hours(24);

        // Score should be 0.5 after one half-life
        let score = scorer.calculate_decayed_score(&entry);
        assert!(
            (score - 0.5).abs() < 0.001,
            "Score after one half-life should be ~0.5, got {}",
            score
        );
    }

    #[test]
    fn test_decay_calculation_two_half_lives() {
        let clock = Arc::new(SimulatedClock::new());
        let scorer = Scorer::with_clock(test_config(), clock.clone());
        let entry = create_entry_at_time(&clock);

        // Advance clock by two half-lives (48 hours)
        clock.advance_hours(48);

        // Score should be 0.25 after two half-lives
        let score = scorer.calculate_decayed_score(&entry);
        assert!(
            (score - 0.25).abs() < 0.001,
            "Score after two half-lives should be ~0.25, got {}",
            score
        );
    }

    #[test]
    fn test_decay_at_specific_times() {
        let clock = Arc::new(SimulatedClock::new());
        let scorer = Scorer::with_clock(test_config(), clock.clone());
        let entry = create_entry_at_time(&clock);

        // Test decay at 7 days (168 hours = 7 half-lives)
        clock.advance_days(7);
        let score_7d = scorer.calculate_decayed_score(&entry);
        let expected_7d = 0.5_f64.powi(7); // 0.0078125
        assert!(
            (score_7d - expected_7d).abs() < 0.001,
            "Score after 7 days should be ~{:.6}, got {:.6}",
            expected_7d,
            score_7d
        );
    }

    #[test]
    fn test_access_boost_normal() {
        let scorer = Scorer::new(test_config());
        let new_score = scorer.boost_on_access(1.0);
        assert!((new_score - 1.2).abs() < 0.001);
    }

    #[test]
    fn test_access_boost_caps_at_max() {
        let scorer = Scorer::new(test_config());

        // Should cap at max_score
        let capped = scorer.boost_on_access(1.9);
        assert!((capped - 2.0).abs() < 0.001);

        // Even higher should still cap
        let still_capped = scorer.boost_on_access(2.5);
        assert!((still_capped - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_record_access_updates_all_fields() {
        let clock = Arc::new(SimulatedClock::new());
        let scorer = Scorer::with_clock(test_config(), clock.clone());
        let mut entry = create_entry_at_time(&clock);

        // Advance time by 24 hours
        clock.advance_hours(24);

        let old_access_count = entry.access_count;
        let old_last_accessed = entry.last_accessed;

        scorer.record_access(&mut entry);

        assert_eq!(entry.access_count, old_access_count + 1);
        assert!(entry.last_accessed > old_last_accessed);
        // Score should be decayed (0.5) + boost (0.2) = 0.7
        assert!(
            (entry.score - 0.7).abs() < 0.001,
            "Score should be ~0.7, got {}",
            entry.score
        );
    }

    #[test]
    fn test_combined_score() {
        let clock = Arc::new(SimulatedClock::new());
        let scorer = Scorer::with_clock(test_config(), clock.clone());
        let entry = create_entry_at_time(&clock);

        // High similarity, high relevance
        let score = scorer.combined_score(0.9, &entry);
        // 0.9 * 0.7 + 1.0 * 0.3 = 0.63 + 0.3 = 0.93
        assert!((score - 0.93).abs() < 0.001);

        // Low similarity, high relevance
        let score = scorer.combined_score(0.1, &entry);
        // 0.1 * 0.7 + 1.0 * 0.3 = 0.07 + 0.3 = 0.37
        assert!((score - 0.37).abs() < 0.001);
    }

    #[test]
    fn test_should_move_to_cold() {
        let clock = Arc::new(SimulatedClock::new());
        let scorer = Scorer::with_clock(test_config(), clock.clone());
        let entry = create_entry_at_time(&clock);

        // Fresh entry should not move to cold
        assert!(!scorer.should_move_to_cold(&entry, 0.3));

        // Advance 3 half-lives (72 hours), score = 0.125
        clock.advance_hours(72);
        assert!(scorer.should_move_to_cold(&entry, 0.3));
    }

    #[test]
    fn test_should_delete() {
        let clock = Arc::new(SimulatedClock::new());
        let scorer = Scorer::with_clock(test_config(), clock.clone());
        let entry = create_entry_at_time(&clock);

        // Fresh entry should not be deleted
        assert!(!scorer.should_delete(&entry, 0.1));

        // Advance 5 half-lives (120 hours), score = 0.03125
        clock.advance_hours(120);
        assert!(scorer.should_delete(&entry, 0.1));
    }

    #[test]
    fn test_apply_decay() {
        let clock = Arc::new(SimulatedClock::new());
        let scorer = Scorer::with_clock(test_config(), clock.clone());
        let mut entry = create_entry_at_time(&clock);

        clock.advance_hours(24);
        let old_last_accessed = entry.last_accessed;
        scorer.apply_decay(&mut entry);

        assert!((entry.score - 0.5).abs() < 0.001);
        // apply_decay should also update last_accessed to prevent double-decay
        assert!(entry.last_accessed > old_last_accessed);
    }

    #[test]
    fn test_active_recall_restores_decayed_memory() {
        let clock = Arc::new(SimulatedClock::new());
        let scorer = Scorer::with_clock(test_config(), clock.clone());
        let mut entry = create_entry_at_time(&clock);

        // Advance 2 half-lives
        clock.advance_hours(48);
        let decayed_score = scorer.calculate_decayed_score(&entry);
        assert!((decayed_score - 0.25).abs() < 0.001);

        // Active recall should boost it
        scorer.record_access(&mut entry);

        // Score should now be decayed + boost = 0.25 + 0.2 = 0.45
        assert!(
            (entry.score - 0.45).abs() < 0.001,
            "Active recall should boost decayed score, got {}",
            entry.score
        );
    }

    #[test]
    fn test_multiple_accesses_accumulate() {
        let clock = Arc::new(SimulatedClock::new());
        let scorer = Scorer::with_clock(test_config(), clock.clone());
        let mut entry = create_entry_at_time(&clock);

        // Multiple accesses should accumulate
        scorer.record_access(&mut entry);
        assert!(
            (entry.score - 1.2).abs() < 0.001,
            "Score after 1 access: {}",
            entry.score
        );

        scorer.record_access(&mut entry);
        assert!(
            (entry.score - 1.4).abs() < 0.001,
            "Score after 2 accesses: {}",
            entry.score
        );

        scorer.record_access(&mut entry);
        assert!(
            (entry.score - 1.6).abs() < 0.001,
            "Score after 3 accesses: {}",
            entry.score
        );

        // Should cap at max
        scorer.record_access(&mut entry);
        scorer.record_access(&mut entry);
        assert!(
            (entry.score - 2.0).abs() < 0.001,
            "Score should cap at max: {}",
            entry.score
        );
    }

    #[test]
    fn test_decay_then_boost_then_decay() {
        // This tests the real-world scenario: memory decays, gets accessed,
        // then decays again from the new higher score
        let clock = Arc::new(SimulatedClock::new());
        let scorer = Scorer::with_clock(test_config(), clock.clone());
        let mut entry = create_entry_at_time(&clock);

        // Day 1: score = 1.0
        assert!((entry.score - 1.0).abs() < 0.001);

        // Day 2: decayed to 0.5, then accessed -> 0.7
        clock.advance_hours(24);
        scorer.record_access(&mut entry);
        assert!(
            (entry.score - 0.7).abs() < 0.001,
            "After access at day 2: {}",
            entry.score
        );

        // Day 3: decays from 0.7 -> 0.35
        clock.advance_hours(24);
        let decayed = scorer.calculate_decayed_score(&entry);
        assert!(
            (decayed - 0.35).abs() < 0.001,
            "After decay at day 3: {}",
            decayed
        );
    }
}
