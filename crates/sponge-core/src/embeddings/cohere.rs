use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use super::{EmbeddingError, EmbeddingProvider};
use crate::constants::{COHERE_DIMENSION, COHERE_LIGHT_DIMENSION};

const COHERE_API_URL: &str = "https://api.cohere.ai/v1/embed";

pub struct CohereProvider {
    client: Client,
    api_key: String,
    model: String,
}

#[derive(Serialize)]
struct CohereRequest<'a> {
    texts: Vec<&'a str>,
    model: &'a str,
    input_type: &'a str,
}

#[derive(Deserialize)]
struct CohereResponse {
    embeddings: Vec<Vec<f32>>,
}

#[derive(Deserialize)]
struct CohereError {
    message: String,
}

impl CohereProvider {
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            model,
        }
    }

    fn dimension_for_model(model: &str) -> usize {
        match model {
            "embed-english-v3.0" => COHERE_DIMENSION,
            "embed-multilingual-v3.0" => COHERE_DIMENSION,
            "embed-english-light-v3.0" => COHERE_LIGHT_DIMENSION,
            "embed-multilingual-light-v3.0" => COHERE_LIGHT_DIMENSION,
            _ => COHERE_DIMENSION,
        }
    }
}

#[async_trait]
impl EmbeddingProvider for CohereProvider {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let results = self.embed_batch(&[text]).await?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| EmbeddingError::InvalidResponse("Empty response".to_string()))
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let request = CohereRequest {
            texts: texts.to_vec(),
            model: &self.model,
            input_type: "search_document",
        };

        let response = self
            .client
            .post(COHERE_API_URL)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            if response.status() == 429 {
                return Err(EmbeddingError::RateLimited(60));
            }
            let error: CohereError = response.json().await.map_err(|e| {
                EmbeddingError::InvalidResponse(format!("Failed to parse error: {}", e))
            })?;
            return Err(EmbeddingError::ApiError(error.message));
        }

        let cohere_response: CohereResponse = response.json().await.map_err(|e| {
            EmbeddingError::InvalidResponse(format!("Failed to parse response: {}", e))
        })?;

        Ok(cohere_response.embeddings)
    }

    fn dimension(&self) -> usize {
        Self::dimension_for_model(&self.model)
    }
}
