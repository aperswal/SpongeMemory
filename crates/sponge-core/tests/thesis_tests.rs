//! Thesis Tests: Proving Active Recall Memory Systems Beat Static Indexes
//!
//! These tests prove the core thesis:
//! "Memory systems that treat retrieval as a read-only operation leave performance on the table.
//!  Retrieval should be a write operation that reshapes the memory landscape."
//!
//! Requirements:
//! - Real Gemini embeddings (no mocking)
//! - Simulated time (injectable clock for instant time advancement)
//! - Never weaken tests - fix bugs in code instead
//! - Run locally with `cargo test --test thesis_tests`
//!
//! Set GOOGLE_API_KEY environment variable or use .env file to run.

mod testdata;

use std::collections::HashSet;
use std::env;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration as StdDuration;

use sponge_core::{
    constants::{
        ACCESS_BOOST_TEST, DECAY_HALF_LIFE_HOURS_TEST, HOT_CAPACITY, MAX_SCORE_TEST,
        SCORE_WEIGHT, SIMILARITY_WEIGHT,
    },
    create_cached_provider, ConsolidationConfig, EmbeddingConfig, Memory, ScoringConfig,
    SimulatedClock, SpongeConfig, StorageConfig,
};
use tempfile::tempdir;
use testdata::{scenarios, TestCorpus};
use uuid::Uuid;

// =============================================================================
// TEST UTILITIES
// =============================================================================

/// Check if verbose output is enabled
fn is_verbose() -> bool {
    env::var("VERBOSE").map(|v| v == "1" || v == "true").unwrap_or(false)
}

/// Check if embedding cache is enabled
fn use_embedding_cache() -> bool {
    env::var("USE_EMBEDDING_CACHE").map(|v| v == "1" || v == "true").unwrap_or(false)
}

/// Get the embedding cache directory
fn cache_dir() -> PathBuf {
    PathBuf::from(".sponge-test-cache")
}

/// Get corpus scale factor (0.0-1.0)
/// Use TEST_CORPUS_SCALE=0.2 for CI, 1.0 for full runs
fn corpus_scale() -> f64 {
    env::var("TEST_CORPUS_SCALE")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(1.0)
        .clamp(0.1, 1.0)
}

/// Scale a corpus count by the configured scale factor
fn scaled_count(base_count: usize) -> usize {
    let scale = corpus_scale();
    (base_count as f64 * scale).max(2.0) as usize
}

/// Load environment from .env file (called once via lazy initialization)
fn init_env() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let _ = dotenvy::from_filename(
            std::env::current_dir()
                .unwrap()
                .join("../../.env")
                .canonicalize()
                .unwrap_or_else(|_| PathBuf::from(".env")),
        );
        let _ = dotenvy::dotenv();
        let _ = dotenvy::from_filename("../../.env");
    });
}

fn get_api_key() -> String {
    init_env();
    env::var("GOOGLE_API_KEY").expect(
        "GOOGLE_API_KEY must be set to run thesis tests. Create a .env file or set the environment variable.",
    )
}

/// Create test config with injectable clock
fn get_test_config(data_path: PathBuf, half_life_hours: f64) -> SpongeConfig {
    let api_key = get_api_key();
    SpongeConfig {
        embedding: EmbeddingConfig::Google {
            api_key,
            model: "gemini-embedding-001".to_string(),
        },
        storage: StorageConfig {
            data_path,
            hot_capacity: HOT_CAPACITY,
            cold_threshold: 0.5, // Intentionally higher than default (0.3) for thesis tests
        },
        scoring: ScoringConfig {
            decay_half_life_hours: half_life_hours,
            access_boost: ACCESS_BOOST_TEST,
            max_score: MAX_SCORE_TEST,
            similarity_weight: SIMILARITY_WEIGHT,
            score_weight: SCORE_WEIGHT,
        },
        consolidation: ConsolidationConfig {
            enabled: false,
            interval_seconds: 3600,
            delete_threshold: 0.1,
        },
        embedding_cache_path: None,
    }
}

/// Rate limit helper - optimized for Gemini's 1500 RPM limit
/// 40ms = 1500 requests/minute theoretical max
async fn rate_limit() {
    tokio::time::sleep(StdDuration::from_millis(45)).await;
}

/// Create memory with optional embedding cache
async fn create_memory_with_config(
    config: SpongeConfig,
    clock: Arc<SimulatedClock>,
) -> Memory {
    if use_embedding_cache() {
        if is_verbose() {
            println!("Using embedding cache at {:?}", cache_dir());
        }
        let embedder = Box::new(create_cached_provider(&config.embedding, cache_dir()));
        Memory::with_embedder(config, clock, embedder)
            .await
            .expect("Failed to create memory with cached embedder")
    } else {
        Memory::with_clock(config, clock)
            .await
            .expect("Failed to create memory")
    }
}

/// Check if concurrent insertion is enabled
fn use_concurrent_insert() -> bool {
    env::var("CONCURRENT_INSERT").map(|v| v == "1" || v == "true").unwrap_or(false)
}

/// Batch insert memories with chunking (Gemini batch limit is ~100 items)
/// Returns IDs of inserted memories
/// With CONCURRENT_INSERT=1, sends multiple batches in parallel
async fn batch_insert(
    memory: &Memory,
    user_id: &str,
    contents: &[String],
    chunk_size: usize,
) -> Vec<Uuid> {
    if use_concurrent_insert() && contents.len() > chunk_size {
        // Concurrent mode: send multiple batches in parallel
        use futures::future::join_all;

        let chunks: Vec<Vec<String>> = contents
            .chunks(chunk_size)
            .map(|c| c.to_vec())
            .collect();

        if is_verbose() {
            println!("Concurrent insert: {} batches of {} items", chunks.len(), chunk_size);
        }

        let futures: Vec<_> = chunks.iter().map(|chunk| {
            let refs: Vec<&str> = chunk.iter().map(|s| s.as_str()).collect();
            async move {
                memory.remember_batch(user_id, &refs, None)
                    .await
                    .expect("Batch insert failed")
            }
        }).collect();

        let results = join_all(futures).await;
        results.into_iter().flatten().collect()
    } else {
        // Sequential mode (default)
        let mut all_ids = Vec::with_capacity(contents.len());

        for chunk in contents.chunks(chunk_size) {
            let refs: Vec<&str> = chunk.iter().map(|s| s.as_str()).collect();
            let ids = memory.remember_batch(user_id, &refs, None)
                .await
                .expect("Batch insert failed");
            all_ids.extend(ids);
            rate_limit().await; // Rate limit between batches
        }

        all_ids
    }
}

// =============================================================================
// TEST 1: Active Recall Strength
// =============================================================================
//
// Validates the core mechanism: retrieval strengthens what was retrieved.
// Insert two memories, advance time, retrieve one, compare scores.
// Output: "Retrieved memories are Nx stronger than forgotten ones."

#[tokio::test]
async fn test_active_recall_strength() {
    println!("\n{}", "=".repeat(60));
    println!("TEST 1: Active Recall Strength");
    println!("{}\n", "=".repeat(60));

    let dir = tempdir().unwrap();
    let clock = Arc::new(SimulatedClock::new());
    let config = get_test_config(dir.path().to_path_buf(), DECAY_HALF_LIFE_HOURS_TEST);

    let memory = create_memory_with_config(config, clock.clone()).await;

    let user_id = "test_user";

    // Load corpus for realistic content
    let corpus = TestCorpus::load();
    let sample = corpus.sample(2, 42);

    // Insert two memories with SAME content (isolates active recall effect)
    let identical_content = &sample[0].content;

    let retrieved_id = memory
        .remember(user_id, identical_content, None)
        .await
        .expect("Failed to store");
    rate_limit().await;

    let forgotten_id = memory
        .remember(user_id, identical_content, None)
        .await
        .expect("Failed to store");
    rate_limit().await;

    if is_verbose() {
        println!("Stored 2 identical memories, advancing 7 days...");
        println!("Content: {}...", &identical_content[..80.min(identical_content.len())]);
    }

    // Advance 7 days - both memories decay equally
    clock.advance_days(7);

    // Recall - only one will be returned and boosted
    let _ = memory
        .recall(user_id, identical_content, 1)
        .await
        .expect("Recall failed");

    // Compare scores - both had identical content and decay, only difference is access
    let retrieved_score = memory
        .get_with_decayed_score(&retrieved_id)
        .map(|(_, s)| s)
        .unwrap();
    let forgotten_score = memory
        .get_with_decayed_score(&forgotten_id)
        .map(|(_, s)| s)
        .unwrap();

    // Since both are identical, the one returned first gets the boost
    let (boosted_score, unboosted_score) = if retrieved_score > forgotten_score {
        (retrieved_score, forgotten_score)
    } else {
        (forgotten_score, retrieved_score)
    };

    let strength_ratio = boosted_score / unboosted_score;

    println!("RESULT: Retrieved memories are {:.0}x stronger than forgotten ones", strength_ratio);
    println!("  Boosted score:   {:.6}", boosted_score);
    println!("  Unboosted score: {:.6}", unboosted_score);

    // Hard assertion: retrieved must be significantly stronger
    assert!(
        strength_ratio >= 10.0,
        "Retrieved memory should be at least 10x stronger, got {:.1}x",
        strength_ratio
    );

    println!("\n[PASSED] Active recall creates {:.0}x differentiation\n", strength_ratio);
}

// =============================================================================
// TEST 2: Precision Improvement Over Time (Hero Metric)
// =============================================================================
//
// The primary metric for the landing page.
// Create mixed corpus, measure precision@10 before and after training.
// Output: "Precision improved from X% to Y% after one week of use."

#[tokio::test]
async fn test_precision_improvement_over_time() {
    println!("\n{}", "=".repeat(60));
    println!("TEST 2: Precision Improvement Over Time");
    println!("{}\n", "=".repeat(60));

    let dir = tempdir().unwrap();
    let clock = Arc::new(SimulatedClock::new());
    let config = get_test_config(dir.path().to_path_buf(), DECAY_HALF_LIFE_HOURS_TEST);

    let memory = create_memory_with_config(config, clock.clone()).await;

    let user_id = "test_user";

    // Load realistic test data from corpus
    let corpus = TestCorpus::load();
    let work_count = scaled_count(50);
    let noise_count = scaled_count(150);
    let scenario = scenarios::precision_test(&corpus, work_count, noise_count);

    // Batch insert work memories
    if is_verbose() {
        println!("Batch inserting {} work + {} noise memories (scale={:.1})...",
                 scenario.work_memories.len(), scenario.noise_memories.len(), corpus_scale());
    }

    let work_ids: HashSet<Uuid> = batch_insert(&memory, user_id, &scenario.work_memories, 50)
        .await
        .into_iter()
        .collect();

    let _ = batch_insert(&memory, user_id, &scenario.noise_memories, 50).await;

    println!("Corpus: {} work + {} noise = {} memories (batch inserted)",
             scenario.work_memories.len(), scenario.noise_memories.len(),
             scenario.work_memories.len() + scenario.noise_memories.len());

    // Measure Day 1 baseline precision@10 and MRR (use readonly to not contaminate state)
    let eval_query = &scenario.eval_queries[0];
    let day1_results = memory.recall_readonly(user_id, eval_query, 10).await.expect("Recall failed");
    rate_limit().await;

    let day1_work_count = day1_results.iter()
        .filter(|r| work_ids.contains(&r.entry.id))
        .count();
    let day1_precision = (day1_work_count as f64 / 10.0) * 100.0;

    // MRR: 1/position of first work result
    let day1_mrr = day1_results.iter()
        .position(|r| work_ids.contains(&r.entry.id))
        .map(|pos| 1.0 / (pos + 1) as f64)
        .unwrap_or(0.0) * 100.0;

    println!("Day 1 Precision@10: {:.0}% ({}/10 work), MRR: {:.0}%", day1_precision, day1_work_count, day1_mrr);

    if is_verbose() {
        println!("  Eval query: {}", eval_query);
        println!("  Top 3 results:");
        for (i, r) in day1_results.iter().take(3).enumerate() {
            let is_work = if work_ids.contains(&r.entry.id) { "[WORK]" } else { "[NOISE]" };
            println!("    {}. {} sim={:.3} score={:.3}: {}...",
                     i + 1, is_work, r.similarity, r.combined_score,
                     &r.entry.content[..60.min(r.entry.content.len())]);
        }
    }

    // Simulate 7 days of training
    if is_verbose() {
        println!("\nTraining: 7 days × {} queries/day...", scenario.training_queries.len());
    }

    for day in 1..=7 {
        for query in &scenario.training_queries {
            let _ = memory.recall(user_id, query, 5).await.expect("Recall failed");
            rate_limit().await;
        }
        clock.advance_days(1);
        memory.consolidate().await;

        if is_verbose() && day % 2 == 0 {
            println!("  Day {} complete", day);
        }
    }

    // Measure Day 7 precision@10 and MRR (same query as Day 1)
    let day7_results = memory.recall(user_id, eval_query, 10).await.expect("Recall failed");

    let day7_work_count = day7_results.iter()
        .filter(|r| work_ids.contains(&r.entry.id))
        .count();
    let day7_precision = (day7_work_count as f64 / 10.0) * 100.0;

    // MRR for Day 7
    let day7_mrr = day7_results.iter()
        .position(|r| work_ids.contains(&r.entry.id))
        .map(|pos| 1.0 / (pos + 1) as f64)
        .unwrap_or(0.0) * 100.0;

    let precision_improvement = day7_precision - day1_precision;
    let mrr_improvement = day7_mrr - day1_mrr;

    println!("Day 7 Precision@10: {:.0}% ({}/10 work), MRR: {:.0}%", day7_precision, day7_work_count, day7_mrr);
    println!("\nRESULT:");
    println!("  Precision: {:.0}% → {:.0}% (+{:.0} points)",
             day1_precision, day7_precision, precision_improvement);
    println!("  MRR:       {:.0}% → {:.0}% (+{:.0} points)",
             day1_mrr, day7_mrr, mrr_improvement);

    if is_verbose() {
        println!("  Top 3 Day 7 results:");
        for (i, r) in day7_results.iter().take(3).enumerate() {
            let is_work = if work_ids.contains(&r.entry.id) { "[WORK]" } else { "[NOISE]" };
            println!("    {}. {} sim={:.3} score={:.3}: {}...",
                     i + 1, is_work, r.similarity, r.combined_score,
                     &r.entry.content[..60.min(r.entry.content.len())]);
        }
    }

    // Hard assertion: must improve by at least 20 percentage points
    assert!(
        precision_improvement >= 20.0,
        "Precision must improve by at least 20 points, got {:.1} points",
        precision_improvement
    );

    println!("\n[PASSED] Precision +{:.0} points, MRR +{:.0} points\n", precision_improvement, mrr_improvement);
}

// =============================================================================
// TEST 3: Noise Elimination
// =============================================================================
//
// Proves irrelevant content gets buried.
// After training, noise should not appear in top results.
// Output: "X/20 noise in results (Y% relevance)"

#[tokio::test]
async fn test_noise_elimination() {
    println!("\n{}", "=".repeat(60));
    println!("TEST 3: Noise Elimination");
    println!("{}\n", "=".repeat(60));

    let dir = tempdir().unwrap();
    let clock = Arc::new(SimulatedClock::new());
    let config = get_test_config(dir.path().to_path_buf(), DECAY_HALF_LIFE_HOURS_TEST);

    let memory = create_memory_with_config(config, clock.clone()).await;

    let user_id = "test_user";

    // Load realistic test data
    let corpus = TestCorpus::load();
    let work_count = scaled_count(100);
    let noise_count = scaled_count(100);
    let scenario = scenarios::noise_elimination(&corpus, work_count, noise_count);

    // Batch insert
    if is_verbose() {
        println!("Batch inserting {} work + {} noise memories (scale={:.1})...",
                 scenario.work_memories.len(), scenario.noise_memories.len(), corpus_scale());
    }

    let work_ids: HashSet<Uuid> = batch_insert(&memory, user_id, &scenario.work_memories, 50)
        .await
        .into_iter()
        .collect();

    let noise_ids: HashSet<Uuid> = batch_insert(&memory, user_id, &scenario.noise_memories, 50)
        .await
        .into_iter()
        .collect();

    println!("Corpus: {} work + {} noise = {} memories (batch inserted)",
             scenario.work_memories.len(), scenario.noise_memories.len(),
             scenario.work_memories.len() + scenario.noise_memories.len());

    // Train for 5 days
    if is_verbose() {
        println!("Training: 5 days × {} queries/day...", scenario.training_queries.len());
    }

    for _day in 1..=5 {
        for query in &scenario.training_queries {
            let _ = memory.recall(user_id, query, 5).await.expect("Recall failed");
            rate_limit().await;
        }
        clock.advance_days(1);
        memory.consolidate().await;
    }

    // Evaluate
    let eval_query = &scenario.eval_queries[0];
    let results = memory.recall(user_id, eval_query, 20).await.expect("Recall failed");

    let noise_in_results = results.iter()
        .filter(|r| noise_ids.contains(&r.entry.id))
        .count();

    let relevance_pct = ((20 - noise_in_results) as f64 / 20.0) * 100.0;

    // MRR: position of first work result (lower = better noise elimination)
    let mrr = results.iter()
        .position(|r| work_ids.contains(&r.entry.id))
        .map(|pos| 1.0 / (pos + 1) as f64)
        .unwrap_or(0.0) * 100.0;

    println!("RESULT: {}/20 noise in top-20 ({:.0}% relevance), MRR: {:.0}%",
             noise_in_results, relevance_pct, mrr);

    if is_verbose() {
        println!("\nTop 5 results:");
        for (i, r) in results.iter().take(5).enumerate() {
            let label = if noise_ids.contains(&r.entry.id) { "[NOISE]" } else { "[WORK]" };
            let preview = &r.entry.content[..60.min(r.entry.content.len())];
            println!("  {}. {} {}", i + 1, label, preview);
        }
    }

    // Hard assertion: at least 80% relevance (max 4 noise in top-20)
    assert!(
        noise_in_results <= 4,
        "At most 4 noise memories should appear in top-20, got {}",
        noise_in_results
    );

    println!("\n[PASSED] Only {}/20 noise ({:.0}% relevance), MRR {:.0}%\n", noise_in_results, relevance_pct, mrr);
}

// =============================================================================
// TEST 4: A/B Versus Static Baseline
// =============================================================================
//
// Proves SpongeMemory beats static vector search.
// Compare active recall vs read-only mode on same corpus.
// Output: "SpongeMemory: X% precision. Static: Y% precision."

#[tokio::test]
async fn test_ab_versus_static_baseline() {
    println!("\n{}", "=".repeat(60));
    println!("TEST 4: A/B Versus Static Baseline");
    println!("{}\n", "=".repeat(60));

    let user_id = "test_user";

    // Load realistic test data
    let corpus = TestCorpus::load();
    let work_count = scaled_count(30);
    let noise_count = scaled_count(70);
    let scenario = scenarios::ab_test(&corpus, work_count, noise_count);

    if is_verbose() {
        println!("Corpus: {} work + {} noise (scale={:.1})",
                 scenario.work_memories.len(), scenario.noise_memories.len(), corpus_scale());
    }

    // ===== SYSTEM A: SpongeMemory with Active Recall =====
    let dir_a = tempdir().unwrap();
    let clock_a = Arc::new(SimulatedClock::new());
    let config_a = get_test_config(dir_a.path().to_path_buf(), DECAY_HALF_LIFE_HOURS_TEST);
    let memory_a = create_memory_with_config(config_a, clock_a.clone()).await;

    if is_verbose() {
        println!("Setting up System A (active recall) with batch insert...");
    }

    let work_ids_a: HashSet<Uuid> = batch_insert(&memory_a, user_id, &scenario.work_memories, 50)
        .await
        .into_iter()
        .collect();
    let _ = batch_insert(&memory_a, user_id, &scenario.noise_memories, 50).await;

    // Train System A with training queries
    if is_verbose() {
        println!("Training System A with {} queries...", scenario.training_queries.len() * 10);
    }

    for i in 0..50 {
        let query = &scenario.training_queries[i % scenario.training_queries.len()];
        let results = memory_a.recall(user_id, query, 5).await.expect("Recall failed");

        // Debug: On first iteration, show what training queries retrieve
        if is_verbose() && i == 0 {
            let work_in_results = results.iter().filter(|r| work_ids_a.contains(&r.entry.id)).count();
            println!("  Training query '{}...' retrieved {}/5 work",
                     &query[..40.min(query.len())], work_in_results);
            if work_in_results == 0 {
                println!("  WARNING: Training not boosting work memories!");
            }
        }

        rate_limit().await;

        if i % 10 == 0 {
            clock_a.advance_hours(12);
            memory_a.consolidate().await;
        }
    }

    // ===== SYSTEM B: Static Mode (same corpus, read-only recall) =====
    let dir_b = tempdir().unwrap();
    let clock_b = Arc::new(SimulatedClock::new());
    let config_b = get_test_config(dir_b.path().to_path_buf(), DECAY_HALF_LIFE_HOURS_TEST);
    let memory_b = create_memory_with_config(config_b, clock_b.clone()).await;

    if is_verbose() {
        println!("Setting up System B (static/read-only) with batch insert...");
    }

    let work_ids_b: HashSet<Uuid> = batch_insert(&memory_b, user_id, &scenario.work_memories, 50)
        .await
        .into_iter()
        .collect();
    let _ = batch_insert(&memory_b, user_id, &scenario.noise_memories, 50).await;

    // "Train" System B with read-only queries (simulates user queries that don't learn)
    if is_verbose() {
        println!("System B: 50 read-only queries (no learning)...");
    }
    for i in 0..50 {
        let query = &scenario.training_queries[i % scenario.training_queries.len()];
        // Use recall_readonly - no score boosting
        let _ = memory_b.recall_readonly(user_id, query, 5).await.expect("Recall failed");
        rate_limit().await;

        if i % 10 == 0 {
            clock_b.advance_hours(12);
        }
    }

    // Evaluate both systems using Mean Reciprocal Rank (MRR)
    // MRR measures WHERE the first relevant result appears - better captures ranking quality
    let mut mrr_a_total = 0.0;
    let mut mrr_b_total = 0.0;

    if is_verbose() {
        println!("Evaluating both systems with {} queries (using MRR)...", scenario.eval_queries.len() * 20);
    }

    for i in 0..20 {
        let query = &scenario.eval_queries[i % scenario.eval_queries.len()];

        // System A (trained with active recall) - use recall_readonly for eval
        // to measure final state without further boosting
        let results_a = memory_a.recall_readonly(user_id, query, 10).await.expect("Recall failed");

        // Calculate MRR: 1/position of first work result
        let first_work_pos_a = results_a.iter()
            .position(|r| work_ids_a.contains(&r.entry.id))
            .map(|pos| 1.0 / (pos + 1) as f64)
            .unwrap_or(0.0);
        mrr_a_total += first_work_pos_a;

        // Debug: On first iteration, show what eval query retrieves
        if is_verbose() && i == 0 {
            let work_a = results_a.iter().filter(|r| work_ids_a.contains(&r.entry.id)).count();
            println!("  Eval query '{}...' retrieved {}/10 work (System A)",
                     &query[..40.min(query.len())], work_a);
            println!("  Top 5 results:");
            for (j, r) in results_a.iter().take(5).enumerate() {
                let label = if work_ids_a.contains(&r.entry.id) { "[WORK]" } else { "[NOISE]" };
                println!("    {}. {} sim={:.3} score={:.3}: {}...",
                         j + 1, label, r.similarity, r.combined_score,
                         &r.entry.content[..50.min(r.entry.content.len())]);
            }
        }

        rate_limit().await;

        // System B (static) - use recall_readonly for pure similarity ranking
        let results_b = memory_b.recall_readonly(user_id, query, 10).await.expect("Recall failed");

        // Calculate MRR for System B
        let first_work_pos_b = results_b.iter()
            .position(|r| work_ids_b.contains(&r.entry.id))
            .map(|pos| 1.0 / (pos + 1) as f64)
            .unwrap_or(0.0);
        mrr_b_total += first_work_pos_b;

        // Debug: On first iteration, show System B results
        if is_verbose() && i == 0 {
            let work_b = results_b.iter().filter(|r| work_ids_b.contains(&r.entry.id)).count();
            println!("  Eval query '{}...' retrieved {}/10 work (System B)",
                     &query[..40.min(query.len())], work_b);
            println!("  System B Top 5:");
            for (j, r) in results_b.iter().take(5).enumerate() {
                let label = if work_ids_b.contains(&r.entry.id) { "[WORK]" } else { "[NOISE]" };
                println!("    {}. {} sim={:.3} score={:.3}", j + 1, label, r.similarity, r.combined_score);
            }
        }

        rate_limit().await;
    }

    // MRR as percentage (max 100% = first result is always work)
    let mrr_a = (mrr_a_total / 20.0) * 100.0;
    let mrr_b = (mrr_b_total / 20.0) * 100.0;
    let improvement = mrr_a - mrr_b;
    let relative_improvement = if mrr_b > 0.0 { (improvement / mrr_b) * 100.0 } else { 0.0 };

    println!("RESULT (Mean Reciprocal Rank):");
    println!("  SpongeMemory (active recall): {:.1}% MRR", mrr_a);
    println!("  Static search (no learning):  {:.1}% MRR", mrr_b);
    println!("  Improvement: +{:.1} points ({:.0}% relative)", improvement, relative_improvement);

    // Hard assertion: SpongeMemory must beat static by at least 15 MRR points
    // MRR = 100% means work at position 1, MRR = 33% means work at position 3
    // So 15 points = roughly moving work up 1-2 positions
    assert!(
        improvement >= 15.0,
        "SpongeMemory must beat static by at least 15 MRR points, got {:.1}",
        improvement
    );

    println!("\n[PASSED] SpongeMemory beats static by {:.1} MRR points\n", improvement);
}

// =============================================================================
// TEST 5: Learning Speed
// =============================================================================
//
// Measures how quickly the system learns user preferences.
// Output: "System learns your priorities in N queries."

#[tokio::test]
async fn test_learning_speed() {
    println!("\n{}", "=".repeat(60));
    println!("TEST 5: Learning Speed");
    println!("{}\n", "=".repeat(60));

    let dir = tempdir().unwrap();
    let clock = Arc::new(SimulatedClock::new());
    let config = get_test_config(dir.path().to_path_buf(), DECAY_HALF_LIFE_HOURS_TEST);

    let memory = create_memory_with_config(config, clock.clone()).await;

    let user_id = "test_user";

    // Load realistic test data
    let corpus = TestCorpus::load();
    let target_count = scaled_count(40);
    let other_count = scaled_count(60);
    let scenario = scenarios::learning_speed(&corpus, target_count, other_count);

    // Batch insert
    if is_verbose() {
        println!("Batch inserting {} target + {} other memories (scale={:.1})",
                 scenario.target_memories.len(), scenario.other_memories.len(), corpus_scale());
    }

    let target_ids: HashSet<Uuid> = batch_insert(&memory, user_id, &scenario.target_memories, 50)
        .await
        .into_iter()
        .collect();
    let _ = batch_insert(&memory, user_id, &scenario.other_memories, 50).await;

    let total_count = scenario.target_memories.len() + scenario.other_memories.len();
    println!("Corpus: {} memories ({} target, batch inserted)", total_count, scenario.target_memories.len());

    // Target: 80% precision@10 on target queries
    let target_precision = 80.0;
    let eval_query = &scenario.eval_query;

    // Run queries one at a time, measuring precision after each
    let mut queries_to_threshold = 0;
    let max_queries = 75;

    for i in 0..max_queries {
        // Training query
        let query = &scenario.training_queries[i % scenario.training_queries.len()];
        let train_results = memory.recall(user_id, query, 5).await.expect("Recall failed");

        // Debug: On first iteration, show what training retrieves
        if is_verbose() && i == 0 {
            let target_in_train = train_results.iter().filter(|r| target_ids.contains(&r.entry.id)).count();
            println!("  Training query '{}...' retrieved {}/5 target",
                     &query[..40.min(query.len())], target_in_train);
        }

        rate_limit().await;

        // Occasionally advance time
        if i % 5 == 0 {
            clock.advance_hours(4);
        }

        // Measure precision
        let results = memory.recall(user_id, eval_query, 10).await.expect("Recall failed");
        rate_limit().await;

        let target_count = results.iter().filter(|r| target_ids.contains(&r.entry.id)).count();
        let precision = (target_count as f64 / 10.0) * 100.0;

        if is_verbose() && i % 10 == 0 {
            println!("  Query {}: precision = {:.0}%", i + 1, precision);
            // Debug: Show why precision isn't improving
            if i == 0 {
                println!("  Eval query '{}...' top 3:", &eval_query[..40.min(eval_query.len())]);
                for (j, r) in results.iter().take(3).enumerate() {
                    let label = if target_ids.contains(&r.entry.id) { "[TARGET]" } else { "[OTHER]" };
                    println!("    {}. {} sim={:.3} score={:.3}", j + 1, label, r.similarity, r.combined_score);
                }
            }
        }

        if precision >= target_precision {
            queries_to_threshold = i + 1;
            break;
        }
    }

    if queries_to_threshold > 0 {
        println!("RESULT: System learns your priorities in {} queries", queries_to_threshold);

        // At 3 queries/day, this is ~N/3 days to tune
        let days_to_tune = (queries_to_threshold as f64 / 3.0).ceil() as i32;
        println!("  At 3 queries/day = ~{} days of normal use", days_to_tune);

        // Hard assertion: must reach 80% in under 75 queries
        assert!(
            queries_to_threshold <= 75,
            "Must reach 80% precision in under 75 queries, took {}",
            queries_to_threshold
        );

        println!("\n[PASSED] Learned in {} queries\n", queries_to_threshold);
    } else {
        println!("RESULT: Did not reach {:.0}% precision in {} queries", target_precision, max_queries);
        panic!("Failed to reach target precision in {} queries", max_queries);
    }
}

// =============================================================================
// TEST 6: Latency Overhead
// =============================================================================
//
// Proves active recall doesn't kill performance.
// Measures write-back overhead vs read-only baseline by comparing recall() vs recall_readonly().
// Output: "Active recall adds Xms per query (Y% overhead)."

#[tokio::test]
async fn test_latency_overhead() {
    println!("\n{}", "=".repeat(60));
    println!("TEST 6: Latency Overhead");
    println!("{}\n", "=".repeat(60));

    let dir = tempdir().unwrap();
    let clock = Arc::new(SimulatedClock::new());
    let config = get_test_config(dir.path().to_path_buf(), DECAY_HALF_LIFE_HOURS_TEST);

    let memory = create_memory_with_config(config, clock.clone()).await;

    let user_id = "test_user";

    // Load corpus for realistic content
    let corpus = TestCorpus::load();
    let mem_count = scaled_count(100);
    let memories: Vec<String> = corpus.sample(mem_count, 123)
        .into_iter()
        .map(|m| m.content.clone())
        .collect();

    if is_verbose() {
        println!("Batch inserting {} memories (scale={:.1})...", memories.len(), corpus_scale());
    }

    let _ = batch_insert(&memory, user_id, &memories, 50).await;

    println!("Corpus: {} memories (batch inserted)", memories.len());

    let queries = [
        "authentication implementation",
        "database optimization",
        "API endpoint design",
        "microservice architecture",
        "payment processing",
    ];

    // Measure read-only latencies (baseline)
    let mut readonly_latencies: Vec<std::time::Duration> = Vec::new();

    if is_verbose() {
        println!("Running 25 read-only queries (baseline)...");
    }

    for i in 0..25 {
        let query = queries[i % queries.len()];

        let start = std::time::Instant::now();
        let _ = memory.recall_readonly(user_id, query, 10).await.expect("Recall failed");
        readonly_latencies.push(start.elapsed());

        rate_limit().await;
    }

    // Measure write-back latencies (active recall)
    let mut writeback_latencies: Vec<std::time::Duration> = Vec::new();

    if is_verbose() {
        println!("Running 25 write-back queries (active recall)...");
    }

    for i in 0..25 {
        let query = queries[i % queries.len()];

        let start = std::time::Instant::now();
        let _ = memory.recall(user_id, query, 10).await.expect("Recall failed");
        writeback_latencies.push(start.elapsed());

        rate_limit().await;
    }

    // Calculate percentiles for both
    readonly_latencies.sort();
    writeback_latencies.sort();

    let readonly_p50 = readonly_latencies[readonly_latencies.len() / 2];
    let readonly_p99 = readonly_latencies[(readonly_latencies.len() * 99) / 100];
    let readonly_avg = readonly_latencies.iter().sum::<std::time::Duration>() / readonly_latencies.len() as u32;

    let writeback_p50 = writeback_latencies[writeback_latencies.len() / 2];
    let writeback_p99 = writeback_latencies[(writeback_latencies.len() * 99) / 100];
    let writeback_avg = writeback_latencies.iter().sum::<std::time::Duration>() / writeback_latencies.len() as u32;

    // Calculate overhead
    let overhead_ms = writeback_avg.saturating_sub(readonly_avg).as_millis();
    let overhead_pct = if readonly_avg.as_millis() > 0 {
        ((writeback_avg.as_millis() as f64 - readonly_avg.as_millis() as f64) / readonly_avg.as_millis() as f64) * 100.0
    } else {
        0.0
    };

    println!("RESULT: Query latencies comparison");
    println!("  Read-only (baseline):");
    println!("    P50: {:?}", readonly_p50);
    println!("    P99: {:?}", readonly_p99);
    println!("    Avg: {:?}", readonly_avg);
    println!("  Write-back (active recall):");
    println!("    P50: {:?}", writeback_p50);
    println!("    P99: {:?}", writeback_p99);
    println!("    Avg: {:?}", writeback_avg);
    println!();
    println!("  Overhead: {}ms ({:.1}%)", overhead_ms, overhead_pct);

    // Hard assertion: P99 must be under 500ms (reasonable for API call + processing)
    assert!(
        writeback_p99 < std::time::Duration::from_millis(500),
        "P99 latency must be under 500ms, got {:?}",
        writeback_p99
    );

    // Soft check: overhead should be minimal
    if overhead_ms < 10 {
        println!("\n[PASSED] Write-back overhead is minimal (<10ms)");
    } else {
        println!("\n[PASSED] Total latency acceptable (P99 < 500ms)");
    }
    println!();
}
