use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use super::{EmbeddingError, EmbeddingProvider};
use crate::constants::{VOYAGE_DIMENSION, VOYAGE_LITE_DIMENSION};

const VOYAGE_API_URL: &str = "https://api.voyageai.com/v1/embeddings";

pub struct VoyageProvider {
    client: Client,
    api_key: String,
    model: String,
}

#[derive(Serialize)]
struct VoyageRequest<'a> {
    input: Vec<&'a str>,
    model: &'a str,
    input_type: &'a str,
}

#[derive(Deserialize)]
struct VoyageResponse {
    data: Vec<VoyageEmbedding>,
}

#[derive(Deserialize)]
struct VoyageEmbedding {
    embedding: Vec<f32>,
}

#[derive(Deserialize)]
struct VoyageError {
    detail: String,
}

impl VoyageProvider {
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            model,
        }
    }

    fn dimension_for_model(model: &str) -> usize {
        match model {
            "voyage-3" => VOYAGE_DIMENSION,
            "voyage-3-lite" => VOYAGE_LITE_DIMENSION,
            "voyage-code-3" => VOYAGE_DIMENSION,
            "voyage-finance-2" => VOYAGE_DIMENSION,
            "voyage-law-2" => VOYAGE_DIMENSION,
            _ => VOYAGE_DIMENSION,
        }
    }
}

#[async_trait]
impl EmbeddingProvider for VoyageProvider {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let results = self.embed_batch(&[text]).await?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| EmbeddingError::InvalidResponse("Empty response".to_string()))
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let request = VoyageRequest {
            input: texts.to_vec(),
            model: &self.model,
            input_type: "document",
        };

        let response = self
            .client
            .post(VOYAGE_API_URL)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            if response.status() == 429 {
                return Err(EmbeddingError::RateLimited(60));
            }
            let error: VoyageError = response.json().await.map_err(|e| {
                EmbeddingError::InvalidResponse(format!("Failed to parse error: {}", e))
            })?;
            return Err(EmbeddingError::ApiError(error.detail));
        }

        let voyage_response: VoyageResponse = response.json().await.map_err(|e| {
            EmbeddingError::InvalidResponse(format!("Failed to parse response: {}", e))
        })?;

        Ok(voyage_response
            .data
            .into_iter()
            .map(|e| e.embedding)
            .collect())
    }

    fn dimension(&self) -> usize {
        Self::dimension_for_model(&self.model)
    }
}
