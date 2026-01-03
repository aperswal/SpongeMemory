use std::net::SocketAddr;
use std::sync::Arc;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    routing::{delete, get, post},
    Json, Router,
};
use figment::providers::Format;
use serde::{Deserialize, Serialize};
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;
use uuid::Uuid;

use sponge_core::{EmbeddingConfig, Memory, MemoryStats, RecallResult, SpongeConfig};

// === Request/Response Types ===

#[derive(Deserialize)]
struct RememberRequest {
    user_id: String,
    content: String,
    metadata: Option<serde_json::Value>,
}

#[derive(Serialize)]
struct RememberResponse {
    id: Uuid,
}

#[derive(Deserialize)]
struct RecallRequest {
    user_id: String,
    query: String,
    #[serde(default = "default_limit")]
    limit: usize,
}

fn default_limit() -> usize {
    10
}

#[derive(Serialize)]
struct RecallResponse {
    results: Vec<RecallResultResponse>,
}

#[derive(Serialize)]
struct RecallResultResponse {
    id: Uuid,
    content: String,
    similarity: f32,
    combined_score: f64,
    metadata: Option<serde_json::Value>,
}

impl From<RecallResult> for RecallResultResponse {
    fn from(r: RecallResult) -> Self {
        Self {
            id: r.entry.id,
            content: r.entry.content,
            similarity: r.similarity,
            combined_score: r.combined_score,
            metadata: r.entry.metadata,
        }
    }
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

#[derive(Serialize)]
struct StatsResponse {
    stats: MemoryStats,
}

#[derive(Serialize)]
struct DeleteResponse {
    deleted: bool,
}

#[derive(Serialize)]
struct DeleteUserResponse {
    deleted_count: usize,
}

// === App State ===

struct AppState {
    memory: Memory,
}

// === Handlers ===

async fn health() -> &'static str {
    "ok"
}

async fn remember(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RememberRequest>,
) -> Result<Json<RememberResponse>, (StatusCode, Json<ErrorResponse>)> {
    match state
        .memory
        .remember(&req.user_id, &req.content, req.metadata)
        .await
    {
        Ok(id) => Ok(Json(RememberResponse { id })),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )),
    }
}

async fn recall(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RecallRequest>,
) -> Result<Json<RecallResponse>, (StatusCode, Json<ErrorResponse>)> {
    match state
        .memory
        .recall(&req.user_id, &req.query, req.limit)
        .await
    {
        Ok(results) => Ok(Json(RecallResponse {
            results: results.into_iter().map(Into::into).collect(),
        })),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )),
    }
}

async fn get_memory(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<Json<sponge_core::MemoryEntry>, StatusCode> {
    state.memory.get(&id).map(Json).ok_or(StatusCode::NOT_FOUND)
}

async fn forget(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Json<DeleteResponse> {
    let deleted = state.memory.forget(&id);
    Json(DeleteResponse { deleted })
}

async fn forget_user(
    State(state): State<Arc<AppState>>,
    Path(user_id): Path<String>,
) -> Json<DeleteUserResponse> {
    let deleted_count = state.memory.forget_user(&user_id);
    Json(DeleteUserResponse { deleted_count })
}

async fn stats(State(state): State<Arc<AppState>>) -> Json<StatsResponse> {
    Json(StatsResponse {
        stats: state.memory.stats(),
    })
}

async fn consolidate(State(state): State<Arc<AppState>>) -> &'static str {
    state.memory.consolidate().await;
    "ok"
}

// === Server Config ===

#[derive(Deserialize)]
struct ServerConfig {
    host: String,
    port: u16,
    #[serde(flatten)]
    sponge: SpongeConfig,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8080,
            sponge: SpongeConfig::default(),
        }
    }
}

// === Main ===

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    // Load config from environment and file
    let config: ServerConfig = figment::Figment::new()
        .merge(figment::providers::Env::prefixed("SPONGE_").split("_"))
        .merge(figment::providers::Toml::file("sponge.toml"))
        .extract()
        .unwrap_or_else(|e| {
            eprintln!("Config error: {}", e);
            eprintln!("Using defaults with environment variables");

            // Build config from individual env vars
            let embedding = if let Ok(key) = std::env::var("VOYAGE_API_KEY") {
                EmbeddingConfig::Voyage {
                    api_key: key,
                    model: std::env::var("VOYAGE_MODEL").unwrap_or_else(|_| "voyage-3".to_string()),
                }
            } else if let Ok(key) = std::env::var("OPENAI_API_KEY") {
                EmbeddingConfig::OpenAI {
                    api_key: key,
                    model: std::env::var("OPENAI_MODEL")
                        .unwrap_or_else(|_| "text-embedding-3-large".to_string()),
                }
            } else if let Ok(key) = std::env::var("COHERE_API_KEY") {
                EmbeddingConfig::Cohere {
                    api_key: key,
                    model: std::env::var("COHERE_MODEL")
                        .unwrap_or_else(|_| "embed-english-v3.0".to_string()),
                }
            } else if let Ok(key) = std::env::var("GOOGLE_API_KEY") {
                EmbeddingConfig::Google {
                    api_key: key,
                    model: std::env::var("GOOGLE_MODEL")
                        .unwrap_or_else(|_| "text-embedding-004".to_string()),
                }
            } else {
                panic!("No embedding API key provided. Set one of: VOYAGE_API_KEY, OPENAI_API_KEY, COHERE_API_KEY, GOOGLE_API_KEY");
            };

            ServerConfig {
                host: std::env::var("HOST").unwrap_or_else(|_| "0.0.0.0".to_string()),
                port: std::env::var("PORT")
                    .ok()
                    .and_then(|p| p.parse().ok())
                    .unwrap_or(8080),
                sponge: SpongeConfig {
                    embedding,
                    ..Default::default()
                },
            }
        });

    info!("Starting Sponge server on {}:{}", config.host, config.port);

    // Initialize memory system
    let memory = Memory::new(config.sponge).await?;
    let state = Arc::new(AppState { memory });

    // Build router
    let app = Router::new()
        .route("/health", get(health))
        .route("/remember", post(remember))
        .route("/recall", post(recall))
        .route("/memory/{id}", get(get_memory))
        .route("/memory/{id}", delete(forget))
        .route("/user/{user_id}", delete(forget_user))
        .route("/stats", get(stats))
        .route("/consolidate", post(consolidate))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    // Run server
    let addr: SocketAddr = format!("{}:{}", config.host, config.port).parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    info!("Sponge server listening on {}", addr);

    axum::serve(listener, app).await?;

    Ok(())
}
