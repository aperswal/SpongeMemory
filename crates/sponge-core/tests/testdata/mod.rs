//! Test Data Loader
//!
//! Loads pre-generated memory corpus for thesis tests.
//! The corpus contains ~1500 realistic work memories across 5 personas and 30 topics.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

/// A persona with associated topics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Persona {
    pub id: String,
    pub name: String,
    pub role: String,
    pub description: String,
    pub topics: Vec<String>,
}

/// A single memory entry from the corpus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub content: String,
    pub topic: String,
    pub subtopic: String,
    pub persona_id: String,
}

/// A corpus for a single persona
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryCorpus {
    pub persona: Persona,
    pub memories: Vec<Memory>,
}

/// The complete test corpus with all personas
pub struct TestCorpus {
    pub corpora: Vec<MemoryCorpus>,
    by_persona: HashMap<String, usize>,
    by_topic: HashMap<String, Vec<Memory>>,
}

impl TestCorpus {
    /// Load the corpus from the testdata/corpus directory
    pub fn load() -> Self {
        let corpus_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/testdata/corpus");

        let combined_path = corpus_dir.join("all_personas.json");
        let json = fs::read_to_string(&combined_path)
            .expect("Failed to read corpus file. Run `cargo run --example generate_corpus` first.");

        let corpora: Vec<MemoryCorpus> = serde_json::from_str(&json)
            .expect("Failed to parse corpus JSON");

        let mut by_persona = HashMap::new();
        let mut by_topic: HashMap<String, Vec<Memory>> = HashMap::new();

        for (i, corpus) in corpora.iter().enumerate() {
            by_persona.insert(corpus.persona.id.clone(), i);

            for memory in &corpus.memories {
                by_topic
                    .entry(memory.topic.clone())
                    .or_default()
                    .push(memory.clone());
            }
        }

        Self {
            corpora,
            by_persona,
            by_topic,
        }
    }

    /// Get all memories for a specific persona
    pub fn memories_for_persona(&self, persona_id: &str) -> Vec<&Memory> {
        self.by_persona
            .get(persona_id)
            .map(|&idx| self.corpora[idx].memories.iter().collect())
            .unwrap_or_default()
    }

    /// Get all memories for a specific topic across all personas
    pub fn memories_for_topic(&self, topic: &str) -> Vec<&Memory> {
        self.by_topic
            .get(topic)
            .map(|mems| mems.iter().collect())
            .unwrap_or_default()
    }

    /// Get a random sample of memories (deterministic based on seed)
    pub fn sample(&self, count: usize, seed: u64) -> Vec<&Memory> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let all_memories: Vec<&Memory> = self.corpora
            .iter()
            .flat_map(|c| c.memories.iter())
            .collect();

        if count >= all_memories.len() {
            return all_memories;
        }

        // Simple deterministic sampling using hash-based shuffle
        let mut indices: Vec<usize> = (0..all_memories.len()).collect();
        for i in 0..indices.len() {
            let mut hasher = DefaultHasher::new();
            (seed, i).hash(&mut hasher);
            let j = hasher.finish() as usize % indices.len();
            indices.swap(i, j);
        }

        indices.into_iter()
            .take(count)
            .map(|i| all_memories[i])
            .collect()
    }

    /// Total memory count
    pub fn total_count(&self) -> usize {
        self.corpora.iter().map(|c| c.memories.len()).sum()
    }
}

/// Test scenario configurations
pub mod scenarios {
    use super::*;

    /// Scenario for testing precision improvement
    ///
    /// Key design: Use COMPLETELY DIFFERENT semantic domains for work vs noise.
    /// Work = support_user (customer support memories about orders)
    /// Noise = student (academic memories about homework)
    ///
    /// This ensures:
    /// - Generic eval query matches both domains (all are "user memories")
    /// - Training queries about "orders" only boost work, not noise
    /// - Day 1 has mixed results, Day 7 has work at top
    pub fn precision_test(corpus: &TestCorpus, work_count: usize, noise_count: usize) -> PrecisionScenario {
        // Work: customer support memories (what user actually cares about)
        let work: Vec<String> = corpus
            .memories_for_persona("support_user")
            .into_iter()
            .take(work_count)
            .map(|m| m.content.clone())
            .collect();

        // Noise: student memories (completely different domain)
        let noise: Vec<String> = corpus
            .memories_for_persona("student")
            .into_iter()
            .take(noise_count)
            .map(|m| m.content.clone())
            .collect();

        // GENERIC eval query - matches all "user information" memories
        let eval_query = "What do you remember about me and my history?".to_string();

        PrecisionScenario {
            work_memories: work,
            noise_memories: noise,
            // SPECIFIC training queries - only match support/order content
            training_queries: vec![
                "Tell me about my customer orders and purchases".into(),
                "What support issues have I contacted you about?".into(),
                "Help me with my account and billing".into(),
            ],
            eval_queries: vec![eval_query],
        }
    }

    /// Scenario for A/B test comparing active recall vs static
    ///
    /// Key design: Use support_user vs student (different domains).
    /// Training uses support-specific queries.
    /// Eval uses GENERIC query that matches both.
    ///
    /// This mirrors the precision_test which WORKS - comparing trained state
    /// vs untrained state on a generic query.
    pub fn ab_test(corpus: &TestCorpus, work_count: usize, noise_count: usize) -> PrecisionScenario {
        // Work: support_user memories (what user cares about)
        let work: Vec<String> = corpus
            .memories_for_persona("support_user")
            .into_iter()
            .take(work_count)
            .map(|m| m.content.clone())
            .collect();

        // Noise: student memories (different domain)
        let noise: Vec<String> = corpus
            .memories_for_persona("student")
            .into_iter()
            .take(noise_count)
            .map(|m| m.content.clone())
            .collect();

        // Training queries match work domain (support)
        let training_queries = vec![
            "Tell me about my customer orders and purchases".into(),
            "What support issues have I contacted you about?".into(),
            "Help me with my account and billing".into(),
        ];

        // Eval query is generic - matches both domains
        // This is the same as precision_test which works
        let eval_query = "What do you remember about me and my history?".to_string();

        PrecisionScenario {
            work_memories: work,
            noise_memories: noise,
            training_queries,
            eval_queries: vec![eval_query],
        }
    }

    /// Scenario for noise elimination test
    ///
    /// Key design: Use COMPLETELY DIFFERENT semantic domains for work vs noise.
    /// Work = student (academic memories)
    /// Noise = enterprise_employee (work/HR memories)
    pub fn noise_elimination(corpus: &TestCorpus, work_count: usize, noise_count: usize) -> PrecisionScenario {
        // Work: student academic memories
        let work: Vec<String> = corpus
            .memories_for_persona("student")
            .into_iter()
            .take(work_count)
            .map(|m| m.content.clone())
            .collect();

        // Noise: enterprise employee memories (completely different domain)
        let noise: Vec<String> = corpus
            .memories_for_persona("enterprise_employee")
            .into_iter()
            .take(noise_count)
            .map(|m| m.content.clone())
            .collect();

        // GENERIC eval query - matches all "user information"
        let eval_query = "What do you remember about my daily activities?".to_string();

        PrecisionScenario {
            work_memories: work,
            noise_memories: noise,
            // SPECIFIC training queries - only match student/academic content
            training_queries: vec![
                "Help me with my homework and studying".into(),
                "What classes and exams have I mentioned?".into(),
                "Tell me about my academic progress".into(),
            ],
            eval_queries: vec![eval_query],
        }
    }

    /// Scenario for learning speed test
    ///
    /// Key design: Use COMPLETELY DIFFERENT semantic domains for target vs other.
    /// Target = finance_user (financial memories)
    /// Other = support_user (customer support memories)
    ///
    /// IMPORTANT: Eval query must match target domain so target memories are in
    /// the candidate pool. Uses same semantic space as training.
    pub fn learning_speed(corpus: &TestCorpus, target_count: usize, other_count: usize) -> LearningScenario {
        // Target: finance memories (what user cares about)
        let target: Vec<String> = corpus
            .memories_for_persona("finance_user")
            .into_iter()
            .take(target_count)
            .map(|m| m.content.clone())
            .collect();

        // Other: customer support memories (completely different domain)
        let other: Vec<String> = corpus
            .memories_for_persona("support_user")
            .into_iter()
            .take(other_count)
            .map(|m| m.content.clone())
            .collect();

        // Training and eval use same semantic space - finance-focused queries
        // Eval uses first training query to ensure target is in candidate pool
        let training_queries = vec![
            "Tell me about my investments and savings".into(),
            "What financial goals have I mentioned?".into(),
            "Help me with my budget and money".into(),
        ];

        LearningScenario {
            target_memories: target,
            other_memories: other,
            training_queries: training_queries.clone(),
            // Use first training query as eval - ensures target is in candidate pool
            eval_query: training_queries[0].clone(),
        }
    }
}

/// Configuration for precision-based tests
pub struct PrecisionScenario {
    pub work_memories: Vec<String>,
    pub noise_memories: Vec<String>,
    pub training_queries: Vec<String>,
    pub eval_queries: Vec<String>,
}

/// Configuration for learning speed tests
pub struct LearningScenario {
    pub target_memories: Vec<String>,
    pub other_memories: Vec<String>,
    pub training_queries: Vec<String>,
    pub eval_query: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_corpus_loads() {
        let corpus = TestCorpus::load();
        assert!(corpus.total_count() > 1000, "Should have >1000 memories");
        assert!(corpus.corpora.len() >= 5, "Should have 5 personas");
    }

    #[test]
    fn test_memories_by_persona() {
        let corpus = TestCorpus::load();
        let support = corpus.memories_for_persona("support_user");
        assert!(support.len() >= 200, "Support user should have many memories");
    }

    #[test]
    fn test_memories_by_topic() {
        let corpus = TestCorpus::load();
        let orders = corpus.memories_for_topic("order_issues");
        assert!(orders.len() >= 40, "Should have many order memories");
    }

    #[test]
    fn test_scenario_precision() {
        let corpus = TestCorpus::load();
        let scenario = scenarios::precision_test(&corpus, 50, 100);
        assert_eq!(scenario.work_memories.len(), 50);
        assert!(scenario.noise_memories.len() <= 100);
        assert!(!scenario.training_queries.is_empty());
        assert!(!scenario.eval_queries.is_empty());
    }
}
