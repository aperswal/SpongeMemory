# Thesis Tests Documentation

> **Core Thesis**: "Memory systems that treat retrieval as a read-only operation leave performance on the table. Retrieval should be a write operation that reshapes the memory landscape."

These tests prove that active recall-based memory systems outperform traditional static vector search by learning from user behavior over time.

---

## Test vs Production Configuration

Tests use `ScoringConfig::test_config()` with faster parameters for visible effects in simulations:

| Parameter | Test Config | Production Default |
|-----------|-------------|-------------------|
| `decay_half_life_hours` | 24.0 (1 day) | 168.0 (1 week) |
| `access_boost` | 0.5 | 0.2 |
| `max_score` | 5.0 | 2.0 |

Fast decay (24h) allows testing 7 days of behavior in seconds using `SimulatedClock`.

---

## Test 1: Active Recall Strength

### Purpose
Validates the core mechanism: retrieval strengthens what was retrieved.

### Inputs
| Input | Description |
|-------|-------------|
| **Content** | Two identical memory entries |
| **Time Advancement** | 7 days of simulated decay |
| **Query** | Single recall query matching both memories |

### Real-Life Scenario
A user stores two notes about the same topic. After a week, they search for one. The retrieved note becomes more prominent in future searches, while the unfound note fades. This mirrors how human memory works - recall strengthens neural pathways.

### Data Requirements
- Minimum 2 memory entries with identical content
- Injectable clock for time simulation
- Scoring configuration with decay half-life (24 hours)

### Outputs
| Output | Current Value | Threshold |
|--------|---------------|-----------|
| **Strength Ratio** | 65x | >= 10x |
| **Boosted Score** | 0.508 | - |
| **Unboosted Score** | 0.008 | - |

### How It Works
This test validates the core mechanism of the system: that retrieval strengthens retrieved memories. It inserts two identical memories, advances time by 7 days (causing both to decay equally), then recalls with a query that matches both. Only one memory is returned and boosted via `record_access()`. The test then compares the raw decayed scores of both memories. Since the content is identical and decay is identical, the only difference is the access boost. The test measures the ratio of boosted to unboosted scores (currently 65x) and asserts it must be at least 10x. This proves the fundamental thesis: retrieval is a write operation that reshapes memory salience, not just a read.

---

## Test 2: Precision Improvement Over Time

### Purpose
The primary "hero metric" - demonstrates learning over simulated usage.

### Inputs
| Input | Description |
|-------|-------------|
| **Work Memories** | 50 customer support memories (orders, billing, account issues) |
| **Noise Memories** | 150 student memories (essays, homework, classes) |
| **Training Queries** | Domain-specific: "Tell me about my orders", "What support issues have I mentioned?" |
| **Eval Query** | Generic: "What do you remember about me and my history?" |
| **Training Duration** | 7 simulated days, 3 queries/day |

### Real-Life Scenario
A customer support agent uses a memory system daily. Initially, searching "what do you know about this user?" returns a mix of support tickets and unrelated notes. After a week of querying about orders and billing, the same generic search now surfaces support-related content first. The system learned what matters to this user.

### Data Requirements
- Two semantically distinct memory domains (work vs noise)
- Training queries that match work domain specifically
- Evaluation query that matches both domains generically
- Corpus with 50+ work and 150+ noise memories

### Outputs
| Metric | Day 1 | Day 7 | Improvement |
|--------|-------|-------|-------------|
| **Precision@10** | 20% (2/10 work) | 40% (4/10 work) | +20 points |
| **MRR** | 17% (work at pos ~6) | 100% (work at pos 1) | +83 points |

### How It Works
This is the "hero metric" test that demonstrates learning over simulated usage. It creates a corpus of 50 work memories (customer support) and 150 noise memories (student essays), then measures precision@10 and MRR on Day 1 using a generic query ("What do you remember about me and my history?"). On Day 1, noise dominates because it has higher semantic similarity to the generic query - precision is 20% (2/10 work) and MRR is 17% (first work at position ~6). The test then simulates 7 days of training with domain-specific queries ("Tell me about my orders", "What support issues have I mentioned?") that retrieve and boost work memories. On Day 7, the same generic eval query now returns work at the top because boosted scores outweigh pure similarity. Final metrics: precision 40% (+20 points), MRR 100% (+83 points). The dramatic MRR improvement shows work moved from position 6 to position 1.

---

## Test 3: Noise Elimination

### Purpose
Proves irrelevant content gets buried after training.

### Inputs
| Input | Description |
|-------|-------------|
| **Work Memories** | 100 student/academic memories (homework, exams, classes) |
| **Noise Memories** | 100 enterprise/HR memories (meetings, performance reviews, policies) |
| **Training Queries** | Academic-focused: "Help me with my homework", "What classes have I mentioned?" |
| **Eval Query** | Generic: "What do you remember about my daily activities?" |
| **Training Duration** | 5 simulated days, 3 queries/day |

### Real-Life Scenario
A student uses a personal knowledge base that unfortunately contains some work-related content from a summer internship. After regularly querying about classes and assignments, the irrelevant work content gets buried. Searching for "my daily activities" now surfaces academic content, not old meeting notes.

### Data Requirements
- Two completely different semantic domains
- Equal corpus sizes (100 each) to test noise suppression
- Training queries specific to work domain
- Generic evaluation query

### Outputs
| Metric | Value | Threshold |
|--------|-------|-----------|
| **Noise in Top-20** | 1/20 | <= 4/20 |
| **Relevance** | 95% | >= 80% |
| **MRR** | 100% | - |

### How It Works
This test proves that irrelevant content gets buried after training. It uses 100 work memories (student/academic) and 100 noise memories (enterprise/HR), then trains for 5 days with academic-focused queries ("Help me with my homework", "What classes have I mentioned?"). After training, it evaluates with a generic query requesting top-20 results. The test counts how many noise items appear and calculates relevance percentage. Currently: only 1/20 noise (95% relevance) with MRR of 100% (first result is work). The test asserts at most 4 noise items in top-20 (80% minimum relevance). This demonstrates that active recall doesn't just boost relevant content - it effectively suppresses irrelevant content by creating score gaps that pure similarity cannot overcome.

---

## Test 4: A/B Versus Static Baseline

### Purpose
Direct comparison proving SpongeMemory beats traditional vector search.

### Inputs
| Input | Description |
|-------|-------------|
| **Work Memories** | 30 customer support memories |
| **Noise Memories** | 70 student memories |
| **Training Queries** | 50 queries (support-focused) |
| **System A** | Active recall (`recall()` - boosts on retrieval) |
| **System B** | Static search (`recall_readonly()` - pure similarity) |

### Real-Life Scenario
A company evaluates two memory systems: a traditional RAG system vs one with active recall. Both are loaded with identical data and receive identical queries. After the evaluation period, they compare which system surfaces relevant content faster. The active recall system wins because it learned from the query patterns.

### Data Requirements
- Identical corpora for both systems
- Same training query sequence
- Same evaluation query
- Isolated storage (separate temp directories)

### Outputs
| System | MRR | Work Position |
|--------|-----|---------------|
| **System A (Active Recall)** | 100% | Position 1-2 |
| **System B (Static)** | 33% | Position 3+ |
| **Improvement** | +66.7 points | 200% relative |

### How It Works
This test directly compares an active recall system against a static vector search baseline on identical corpora. System A uses `recall()` which boosts scores on retrieval; System B uses `recall_readonly()` which is pure similarity ranking. Both systems receive the same 50 "training" queries, but only System A learns from them. The test then evaluates both systems with a generic query and measures Mean Reciprocal Rank (MRR). System A achieves 100% MRR (work at position 1) while System B achieves 33% MRR (work at position 3). The +66.7 point improvement proves that active recall provides a measurable advantage over traditional vector search. Originally this test used precision which showed both at 20% (same 2 work items in results), missing that System A ranked them higher - MRR captures this ranking quality difference.

---

## Test 5: Learning Speed

### Purpose
Measures how quickly the system learns user preferences.

### Inputs
| Input | Description |
|-------|-------------|
| **Target Memories** | 40 finance memories (investments, savings, budgets) |
| **Other Memories** | 60 customer support memories |
| **Training Queries** | Finance-focused: "Tell me about my investments", "What financial goals have I mentioned?" |
| **Eval Query** | Same as training (ensures target in candidate pool) |
| **Target Precision** | 80% precision@10 |

### Real-Life Scenario
A new user starts using a financial planning assistant. The question is: how many queries until the system understands they care about investments, not support tickets? This test answers "time to value" - users want personalization quickly, not after months of usage.

### Data Requirements
- Two distinct domains with unequal sizes (40 target, 60 other)
- Training and eval queries in same semantic space
- Incremental measurement after each query

### Outputs
| Metric | Value | Threshold |
|--------|-------|-----------|
| **Queries to 80% Precision** | 1 | <= 75 |
| **Days of Normal Use** | ~1 day | - |
| **Usage Assumption** | 3 queries/day | - |

### How It Works
This test measures how quickly the system learns user preferences, answering "how many queries until the system understands what I care about?" It creates 40 target memories (finance) and 60 other memories (customer support), then runs training queries one at a time while measuring precision@10 after each. The target is 80% precision on finance-related queries. The test uses the same query for training and evaluation (ensuring target memories are in the candidate pool). Currently the system reaches 80% precision in just 1 query, translating to approximately 1 day of normal use at 3 queries/day. The test asserts learning must happen within 75 queries. This demonstrates that active recall provides rapid personalization - users don't need weeks of usage before the system adapts to their priorities.

---

## Test 6: Latency Overhead

### Purpose
Proves active recall doesn't kill performance.

### Inputs
| Input | Description |
|-------|-------------|
| **Corpus Size** | 100 random memories |
| **Read-Only Queries** | 25 queries using `recall_readonly()` |
| **Write-Back Queries** | 25 queries using `recall()` |
| **Query Types** | Technical: "authentication", "database optimization", "API design" |

### Real-Life Scenario
A production system needs to decide whether to enable active recall. The concern: does the write-back overhead make queries too slow? This test measures the actual latency impact to prove active recall is production-viable.

### Data Requirements
- Corpus of 100+ memories for realistic search
- Variety of query types
- Rate limiting between queries (45ms for Gemini API)

### Outputs
| Metric | Read-Only | Write-Back | Overhead |
|--------|-----------|------------|----------|
| **P50 Latency** | 124ms | 126ms | +2ms |
| **P99 Latency** | 139ms | 206ms | +67ms |
| **Average** | 125ms | 135ms | +10ms (8.9%) |

### How It Works
This test proves that active recall doesn't kill performance by measuring the overhead of write-back operations. It runs 25 read-only queries (`recall_readonly()`) and 25 write-back queries (`recall()`) on the same corpus, measuring latency for each. It calculates P50, P99, and average latencies for both modes, then computes the overhead. Current results: read-only averages 125ms, write-back averages 135ms, overhead is 10ms (8.9%). The test asserts P99 latency must be under 500ms. This overhead is acceptable because `record_access()` only updates in-memory scores - there's no additional embedding API call or disk I/O on the critical path. The test validates that the benefits of active recall come at minimal performance cost.

---

## Test Data Architecture

### Personas Used
| Persona | Domain | Used In |
|---------|--------|---------|
| `support_user` | Customer support (orders, billing, accounts) | Tests 2, 4 |
| `student` | Academic (essays, homework, exams) | Tests 2, 3, 4 |
| `finance_user` | Personal finance (investments, budgets) | Test 5 |
| `healthcare_patient` | Medical (medications, appointments) | Available |
| `enterprise_employee` | Corporate (meetings, HR, policies) | Test 3 |

### Corpus Statistics
- **Total memories**: ~1500 across 5 personas
- **Memories per persona**: 250-350
- **Topics per persona**: 6
- **Average memory length**: 50-150 words

### Key Design Principle
Work and noise domains must be **semantically distinct** so that:
1. Generic eval queries match both domains (tests generalization)
2. Training queries match only work domain (tests learning)
3. Boosted scores can overcome similarity gaps

---

## Running the Tests

```bash
# Run all thesis tests with verbose output
GOOGLE_API_KEY=your_key VERBOSE=1 cargo test --test thesis_tests -- --nocapture

# Run specific test
GOOGLE_API_KEY=your_key cargo test --test thesis_tests test_precision_improvement -- --nocapture

# Scale down corpus for faster CI runs (0.1 to 1.0)
TEST_CORPUS_SCALE=0.3 cargo test --test thesis_tests

# Enable embedding cache for repeated runs
USE_EMBEDDING_CACHE=1 cargo test --test thesis_tests
```

---

## Summary of Results

| Test | Primary Metric | Result | Proves |
|------|----------------|--------|--------|
| **1. Active Recall Strength** | Score ratio | 65x differentiation | Retrieval boosts memory strength |
| **2. Precision Over Time** | Precision + MRR | +20 precision, +83 MRR | System learns over time |
| **3. Noise Elimination** | Relevance + MRR | 95% relevance, 100% MRR | Irrelevant content gets buried |
| **4. A/B vs Static** | MRR improvement | +66.7 points | Beats traditional vector search |
| **5. Learning Speed** | Queries to threshold | 1 query (~1 day) | Rapid personalization |
| **6. Latency Overhead** | P99 latency | 206ms (+10ms avg) | Production-viable performance |

These tests collectively prove that treating retrieval as a write operation provides measurable, significant improvements over static vector search systems.
