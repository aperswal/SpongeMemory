//! Injectable clock for time-based operations
//!
//! This module provides a Clock trait that allows the system to use either
//! real time (production) or simulated time (testing). This enables testing
//! time-dependent behavior like decay without waiting real time.

use chrono::{DateTime, Duration, Utc};
use parking_lot::RwLock;
use std::sync::Arc;

/// A clock that provides the current time
///
/// In production, use `SystemClock` which returns real time.
/// In tests, use `SimulatedClock` which can be advanced programmatically.
pub trait Clock: Send + Sync {
    /// Get the current time
    fn now(&self) -> DateTime<Utc>;
}

/// Production clock that returns real system time
#[derive(Clone, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> DateTime<Utc> {
        Utc::now()
    }
}

/// Simulated clock for testing time-dependent behavior
///
/// The clock starts at a fixed point and can be advanced programmatically.
/// This allows testing decay, consolidation, and other time-based features
/// without waiting real time.
#[derive(Clone)]
pub struct SimulatedClock {
    current_time: Arc<RwLock<DateTime<Utc>>>,
}

impl SimulatedClock {
    /// Create a new simulated clock starting at the current real time
    pub fn new() -> Self {
        Self {
            current_time: Arc::new(RwLock::new(Utc::now())),
        }
    }

    /// Create a new simulated clock starting at a specific time
    pub fn starting_at(time: DateTime<Utc>) -> Self {
        Self {
            current_time: Arc::new(RwLock::new(time)),
        }
    }

    /// Advance the clock by a duration
    pub fn advance(&self, duration: Duration) {
        let mut time = self.current_time.write();
        *time += duration;
    }

    /// Advance the clock by a number of days
    pub fn advance_days(&self, days: i64) {
        self.advance(Duration::days(days));
    }

    /// Advance the clock by a number of hours
    pub fn advance_hours(&self, hours: i64) {
        self.advance(Duration::hours(hours));
    }

    /// Set the clock to a specific time
    pub fn set(&self, time: DateTime<Utc>) {
        *self.current_time.write() = time;
    }

    /// Get the current simulated time (for assertions)
    pub fn get(&self) -> DateTime<Utc> {
        *self.current_time.read()
    }
}

impl Default for SimulatedClock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clock for SimulatedClock {
    fn now(&self) -> DateTime<Utc> {
        *self.current_time.read()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_clock_returns_current_time() {
        let clock = SystemClock;
        let before = Utc::now();
        let clock_time = clock.now();
        let after = Utc::now();

        assert!(clock_time >= before);
        assert!(clock_time <= after);
    }

    #[test]
    fn test_simulated_clock_advance_days() {
        let clock = SimulatedClock::new();
        let start = clock.now();

        clock.advance_days(7);

        let after = clock.now();
        let diff = after - start;
        assert_eq!(diff.num_days(), 7);
    }

    #[test]
    fn test_simulated_clock_advance_hours() {
        let clock = SimulatedClock::new();
        let start = clock.now();

        clock.advance_hours(24);

        let after = clock.now();
        let diff = after - start;
        assert_eq!(diff.num_hours(), 24);
    }

    #[test]
    fn test_simulated_clock_set() {
        let clock = SimulatedClock::new();
        let target = Utc::now() + Duration::days(100);

        clock.set(target);

        assert_eq!(clock.now(), target);
    }

    #[test]
    fn test_simulated_clock_is_clone_safe() {
        let clock1 = SimulatedClock::new();
        let clock2 = clock1.clone();

        clock1.advance_days(5);

        // Both clocks should see the same time (Arc shared state)
        assert_eq!(clock1.now(), clock2.now());
    }
}
