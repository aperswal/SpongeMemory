use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::constants::{
    ACCESS_BOOST, ACCESS_BOOST_TEST, COLD_THRESHOLD, CONSOLIDATION_INTERVAL_SECS,
    DECAY_HALF_LIFE_HOURS, DECAY_HALF_LIFE_HOURS_TEST, DELETE_THRESHOLD, HOT_CAPACITY,
    MAX_SCORE, MAX_SCORE_TEST, SCORE_WEIGHT, SIMILARITY_WEIGHT,
};

/// Main configuration for Sponge
#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(Default)]
pub struct SpongeConfig {
    /// Embedding provider configuration
    pub embedding: EmbeddingConfig,
    /// Storage configuration
    pub storage: StorageConfig,
    /// Scoring/decay configuration
    pub scoring: ScoringConfig,
    /// Consolidation configuration
    pub consolidation: ConsolidationConfig,
    /// Optional embedding cache directory. When set, embeddings are cached to disk
    /// to avoid paying for duplicate API calls for the same content.
    #[serde(default)]
    pub embedding_cache_path: Option<PathBuf>,
}


/// Embedding provider configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "provider", rename_all = "lowercase")]
pub enum EmbeddingConfig {
    /// Voyage AI - highest quality embeddings
    Voyage {
        api_key: String,
        #[serde(default = "default_voyage_model")]
        model: String,
    },
    /// OpenAI embeddings
    OpenAI {
        api_key: String,
        #[serde(default = "default_openai_model")]
        model: String,
    },
    /// Cohere embeddings
    Cohere {
        api_key: String,
        #[serde(default = "default_cohere_model")]
        model: String,
    },
    /// Google/Gemini embeddings
    Google {
        api_key: String,
        #[serde(default = "default_google_model")]
        model: String,
    },
}

fn default_voyage_model() -> String {
    "voyage-3".to_string()
}

fn default_openai_model() -> String {
    "text-embedding-3-large".to_string()
}

fn default_cohere_model() -> String {
    "embed-english-v3.0".to_string()
}

fn default_google_model() -> String {
    "gemini-embedding-001".to_string()
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        // Default to OpenAI since most people have keys
        Self::OpenAI {
            api_key: String::new(),
            model: default_openai_model(),
        }
    }
}

/// Storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Path to store persistent data
    pub data_path: PathBuf,
    /// Maximum number of entries in hot storage (LRU eviction)
    pub hot_capacity: usize,
    /// Score threshold below which entries move to cold storage
    pub cold_threshold: f64,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            data_path: PathBuf::from("./sponge-data"),
            hot_capacity: HOT_CAPACITY,
            cold_threshold: COLD_THRESHOLD,
        }
    }
}

/// Scoring and decay configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringConfig {
    /// Half-life for memory decay in hours
    /// Formula: score = initial_score × 0.5^(hours_elapsed / decay_half_life_hours)
    pub decay_half_life_hours: f64,
    /// Boost factor when a memory is accessed (active recall)
    pub access_boost: f64,
    /// Maximum score a memory can have
    pub max_score: f64,
    /// Weight for similarity in combined score calculation (must sum to 1.0 with score_weight)
    pub similarity_weight: f64,
    /// Weight for memory score in combined score calculation (must sum to 1.0 with similarity_weight)
    pub score_weight: f64,
}

impl Default for ScoringConfig {
    fn default() -> Self {
        Self {
            decay_half_life_hours: DECAY_HALF_LIFE_HOURS,
            access_boost: ACCESS_BOOST,
            max_score: MAX_SCORE,
            similarity_weight: SIMILARITY_WEIGHT,
            score_weight: SCORE_WEIGHT,
        }
    }
}

impl ScoringConfig {
    /// Test configuration with faster decay for testing time-dependent behavior
    pub fn test_config() -> Self {
        Self {
            decay_half_life_hours: DECAY_HALF_LIFE_HOURS_TEST,
            access_boost: ACCESS_BOOST_TEST,
            max_score: MAX_SCORE_TEST,
            similarity_weight: SIMILARITY_WEIGHT,
            score_weight: SCORE_WEIGHT,
        }
    }
}

/// Consolidation (background "sleep") configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationConfig {
    /// How often to run consolidation in seconds
    pub interval_seconds: u64,
    /// Score threshold below which memories are deleted
    pub delete_threshold: f64,
    /// Whether to run consolidation automatically
    pub enabled: bool,
}

impl Default for ConsolidationConfig {
    fn default() -> Self {
        Self {
            interval_seconds: CONSOLIDATION_INTERVAL_SECS,
            delete_threshold: DELETE_THRESHOLD,
            enabled: true,
        }
    }
}

/// Configuration validation errors
#[derive(Debug, Clone, PartialEq)]
pub enum ConfigValidationError {
    /// Similarity weight and score weight must sum to 1.0
    WeightsSumInvalid { sum: f64 },
    /// Cold threshold must be greater than delete threshold
    ColdThresholdBelowDelete {
        cold_threshold: f64,
        delete_threshold: f64,
    },
    /// Decay half-life must be positive
    InvalidHalfLife { value: f64 },
    /// Access boost must be non-negative
    InvalidAccessBoost { value: f64 },
    /// Max score must be greater than initial score (1.0)
    InvalidMaxScore { value: f64 },
    /// Weights must be between 0 and 1
    WeightOutOfRange { name: &'static str, value: f64 },
}

impl std::fmt::Display for ConfigValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WeightsSumInvalid { sum } => {
                write!(
                    f,
                    "similarity_weight + score_weight must equal 1.0, got {}",
                    sum
                )
            }
            Self::ColdThresholdBelowDelete {
                cold_threshold,
                delete_threshold,
            } => {
                write!(
                    f,
                    "cold_threshold ({}) must be greater than delete_threshold ({})",
                    cold_threshold, delete_threshold
                )
            }
            Self::InvalidHalfLife { value } => {
                write!(f, "decay_half_life_hours must be positive, got {}", value)
            }
            Self::InvalidAccessBoost { value } => {
                write!(f, "access_boost must be non-negative, got {}", value)
            }
            Self::InvalidMaxScore { value } => {
                write!(
                    f,
                    "max_score must be greater than 1.0 (initial score), got {}",
                    value
                )
            }
            Self::WeightOutOfRange { name, value } => {
                write!(f, "{} must be between 0 and 1, got {}", name, value)
            }
        }
    }
}

impl std::error::Error for ConfigValidationError {}

impl SpongeConfig {
    /// Validate the configuration and return all errors found.
    /// Returns Ok(()) if configuration is valid, or Err with all validation errors.
    pub fn validate(&self) -> Result<(), Vec<ConfigValidationError>> {
        let mut errors = Vec::new();

        // Check similarity_weight + score_weight == 1.0 (with small epsilon for float comparison)
        let weights_sum = self.scoring.similarity_weight + self.scoring.score_weight;
        if (weights_sum - 1.0).abs() > 0.0001 {
            errors.push(ConfigValidationError::WeightsSumInvalid { sum: weights_sum });
        }

        // Check weights are in valid range [0, 1]
        if self.scoring.similarity_weight < 0.0 || self.scoring.similarity_weight > 1.0 {
            errors.push(ConfigValidationError::WeightOutOfRange {
                name: "similarity_weight",
                value: self.scoring.similarity_weight,
            });
        }
        if self.scoring.score_weight < 0.0 || self.scoring.score_weight > 1.0 {
            errors.push(ConfigValidationError::WeightOutOfRange {
                name: "score_weight",
                value: self.scoring.score_weight,
            });
        }

        // Check cold_threshold > delete_threshold
        if self.storage.cold_threshold <= self.consolidation.delete_threshold {
            errors.push(ConfigValidationError::ColdThresholdBelowDelete {
                cold_threshold: self.storage.cold_threshold,
                delete_threshold: self.consolidation.delete_threshold,
            });
        }

        // Check decay_half_life_hours > 0
        if self.scoring.decay_half_life_hours <= 0.0 {
            errors.push(ConfigValidationError::InvalidHalfLife {
                value: self.scoring.decay_half_life_hours,
            });
        }

        // Check access_boost >= 0
        if self.scoring.access_boost < 0.0 {
            errors.push(ConfigValidationError::InvalidAccessBoost {
                value: self.scoring.access_boost,
            });
        }

        // Check max_score > 1.0 (initial score)
        if self.scoring.max_score <= 1.0 {
            errors.push(ConfigValidationError::InvalidMaxScore {
                value: self.scoring.max_score,
            });
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Validate configuration and panic with a clear message if invalid.
    /// Use this at startup to fail fast on misconfiguration.
    pub fn validate_or_panic(&self) {
        if let Err(errors) = self.validate() {
            let error_messages: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
            panic!(
                "Invalid SpongeConfig:\n  - {}",
                error_messages.join("\n  - ")
            );
        }
    }
}
