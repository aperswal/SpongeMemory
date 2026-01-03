//! Corpus Generation Script
//!
//! Generates realistic memory data using Gemini for thesis tests.
//! Run with: cargo run --example generate_corpus
//!
//! This creates test data files in tests/testdata/corpus/

use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::time::Duration;

const GEMINI_API_URL: &str = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Persona {
    pub id: String,
    pub name: String,
    pub role: String,
    pub description: String,
    pub topics: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub content: String,
    pub topic: String,
    pub subtopic: String,
    pub persona_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryCorpus {
    pub persona: Persona,
    pub memories: Vec<Memory>,
}

#[derive(Debug, Serialize)]
struct GeminiRequest {
    contents: Vec<GeminiContent>,
    generation_config: GenerationConfig,
}

#[derive(Debug, Serialize)]
struct GeminiContent {
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Serialize)]
struct GeminiPart {
    text: String,
}

#[derive(Debug, Serialize)]
struct GenerationConfig {
    temperature: f32,
    max_output_tokens: u32,
    response_mime_type: String,
}

#[derive(Debug, Deserialize)]
struct GeminiResponse {
    candidates: Vec<GeminiCandidate>,
}

#[derive(Debug, Deserialize)]
struct GeminiCandidate {
    content: GeminiResponseContent,
}

#[derive(Debug, Deserialize)]
struct GeminiResponseContent {
    parts: Vec<GeminiResponsePart>,
}

#[derive(Debug, Deserialize)]
struct GeminiResponsePart {
    text: String,
}

#[derive(Debug, Deserialize)]
struct GeneratedMemories {
    memories: Vec<GeneratedMemory>,
}

#[derive(Debug, Deserialize)]
struct GeneratedMemory {
    content: String,
    subtopic: String,
}

fn get_personas() -> Vec<Persona> {
    vec![
        // Customer support chatbot user
        Persona {
            id: "support_user".into(),
            name: "Maria Santos".into(),
            role: "Customer Support User".into(),
            description: "Regular customer who contacts support about orders, returns, account issues, and product questions. Gets frustrated repeating information.".into(),
            topics: vec![
                "order_issues".into(),
                "returns_refunds".into(),
                "account_problems".into(),
                "product_questions".into(),
                "shipping_delivery".into(),
                "billing_payments".into(),
            ],
        },
        // Healthcare AI assistant patient
        Persona {
            id: "healthcare_patient".into(),
            name: "Robert Chen".into(),
            role: "Healthcare Patient".into(),
            description: "Patient managing chronic conditions who uses AI health assistant for medication reminders, symptom tracking, appointment scheduling, and health questions.".into(),
            topics: vec![
                "medications".into(),
                "symptoms".into(),
                "appointments".into(),
                "diet_nutrition".into(),
                "exercise_activity".into(),
                "mental_health".into(),
            ],
        },
        // Student using AI tutor
        Persona {
            id: "student".into(),
            name: "Jamie Park".into(),
            role: "College Student".into(),
            description: "Undergraduate student using AI tutor for homework help, exam prep, writing assistance, and learning new concepts across multiple subjects.".into(),
            topics: vec![
                "math_homework".into(),
                "writing_essays".into(),
                "exam_prep".into(),
                "research_projects".into(),
                "study_habits".into(),
                "career_planning".into(),
            ],
        },
        // Financial advisor AI user
        Persona {
            id: "finance_user".into(),
            name: "David Thompson".into(),
            role: "Personal Finance User".into(),
            description: "Working professional using AI financial advisor for budgeting, investment questions, retirement planning, and financial decisions.".into(),
            topics: vec![
                "budgeting".into(),
                "investments".into(),
                "retirement".into(),
                "taxes".into(),
                "major_purchases".into(),
                "debt_management".into(),
            ],
        },
        // Enterprise employee using internal AI assistant
        Persona {
            id: "enterprise_employee".into(),
            name: "Sarah Williams".into(),
            role: "Marketing Manager".into(),
            description: "Mid-level employee using company's internal AI assistant for HR questions, IT help, expense reports, meeting scheduling, and company policies.".into(),
            topics: vec![
                "hr_benefits".into(),
                "it_support".into(),
                "expenses_travel".into(),
                "meetings_calendar".into(),
                "company_policies".into(),
                "team_management".into(),
            ],
        },
    ]
}

async fn generate_memories_for_topic(
    client: &Client,
    api_key: &str,
    persona: &Persona,
    topic: &str,
    count: usize,
) -> Result<Vec<Memory>, Box<dyn std::error::Error>> {
    let prompt = format!(
        r#"You are generating realistic memories that an AI assistant would store about a user over time.

User: {} ({})
Background: {}
Topic: {}

Generate exactly {} unique memories that an AI assistant might store about this user from their conversations.
These are things the AI learned about the user that would be useful to remember for future interactions.

Types of memories to generate (mix these):
- User preferences: "User prefers detailed explanations over quick answers"
- Personal facts: "User mentioned they have two kids in elementary school"
- Past interactions: "User asked about returning a laptop on March 15th - issue was resolved"
- Behavioral patterns: "User typically messages during lunch breaks"
- Stated needs: "User said they're trying to save for a house down payment"
- Context from conversations: "User was frustrated about the shipping delay last week"

Make memories feel like natural things an AI would note from real conversations - not formal profiles.
Include realistic details like dates, specific products, amounts, names of family members, etc.

IMPORTANT: Make memories semantically SIMILAR within the topic - they should overlap enough that a search might return multiple related memories. This tests whether the system can learn which specific memories matter most to the user.

Return as JSON:
{{
  "memories": [
    {{"content": "the memory text", "subtopic": "specific aspect like 'medication_schedule' or 'side_effects' for medications topic"}}
  ]
}}"#,
        persona.name, persona.role, persona.description, topic, count
    );

    let request = GeminiRequest {
        contents: vec![GeminiContent {
            parts: vec![GeminiPart { text: prompt }],
        }],
        generation_config: GenerationConfig {
            temperature: 0.9,
            max_output_tokens: 8192,
            response_mime_type: "application/json".into(),
        },
    };

    let url = format!("{}?key={}", GEMINI_API_URL, api_key);

    let response = client
        .post(&url)
        .json(&request)
        .timeout(Duration::from_secs(60))
        .send()
        .await?;

    if !response.status().is_success() {
        let error_text = response.text().await?;
        return Err(format!("Gemini API error: {}", error_text).into());
    }

    let gemini_response: GeminiResponse = response.json().await?;

    let json_text = &gemini_response.candidates[0].content.parts[0].text;
    let generated: GeneratedMemories = serde_json::from_str(json_text)?;

    Ok(generated
        .memories
        .into_iter()
        .map(|m| Memory {
            content: m.content,
            topic: topic.to_string(),
            subtopic: m.subtopic,
            persona_id: persona.id.clone(),
        })
        .collect())
}

async fn generate_corpus_for_persona(
    client: &Client,
    api_key: &str,
    persona: &Persona,
    memories_per_topic: usize,
) -> Result<MemoryCorpus, Box<dyn std::error::Error>> {
    let mut all_memories = Vec::new();

    for topic in &persona.topics {
        println!("  Generating {} memories for topic '{}'...", memories_per_topic, topic);

        match generate_memories_for_topic(client, api_key, persona, topic, memories_per_topic).await {
            Ok(memories) => {
                println!("    Generated {} memories", memories.len());
                all_memories.extend(memories);
            }
            Err(e) => {
                eprintln!("    Error generating memories for {}/{}: {}", persona.id, topic, e);
            }
        }

        // Rate limit
        tokio::time::sleep(Duration::from_millis(500)).await;
    }

    Ok(MemoryCorpus {
        persona: persona.clone(),
        memories: all_memories,
    })
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load API key
    let _ = dotenvy::dotenv();
    let api_key = std::env::var("GOOGLE_API_KEY")
        .expect("GOOGLE_API_KEY must be set");

    let client = Client::new();
    let personas = get_personas();

    // Create output directory
    let output_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/testdata/corpus");
    fs::create_dir_all(&output_dir)?;

    let memories_per_topic = std::env::var("MEMORIES_PER_TOPIC")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(50);

    println!("Generating corpus with {} memories per topic...", memories_per_topic);
    println!("Output directory: {:?}", output_dir);
    println!();

    let mut all_corpora = Vec::new();

    for persona in &personas {
        println!("Generating corpus for {} ({})...", persona.name, persona.role);

        match generate_corpus_for_persona(&client, &api_key, persona, memories_per_topic).await {
            Ok(corpus) => {
                // Save individual persona file
                let filename = format!("{}.json", persona.id);
                let filepath = output_dir.join(&filename);
                let json = serde_json::to_string_pretty(&corpus)?;
                fs::write(&filepath, &json)?;
                println!("  Saved {} memories to {}", corpus.memories.len(), filename);

                all_corpora.push(corpus);
            }
            Err(e) => {
                eprintln!("  Error generating corpus for {}: {}", persona.id, e);
            }
        }

        println!();
    }

    // Save combined corpus
    let combined_path = output_dir.join("all_personas.json");
    let combined_json = serde_json::to_string_pretty(&all_corpora)?;
    fs::write(&combined_path, &combined_json)?;

    let total_memories: usize = all_corpora.iter().map(|c| c.memories.len()).sum();
    println!("=== Generation Complete ===");
    println!("Total personas: {}", all_corpora.len());
    println!("Total memories: {}", total_memories);
    println!("Saved to: {:?}", output_dir);

    Ok(())
}
