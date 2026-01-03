use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use super::{EmbeddingError, EmbeddingProvider};
use crate::constants::{OPENAI_ADA_DIMENSION, OPENAI_LARGE_DIMENSION, OPENAI_SMALL_DIMENSION};

const OPENAI_API_URL: &str = "https://api.openai.com/v1/embeddings";

pub struct OpenAIProvider {
    client: Client,
    api_key: String,
    model: String,
}

#[derive(Serialize)]
struct OpenAIRequest<'a> {
    input: Vec<&'a str>,
    model: &'a str,
}

#[derive(Deserialize)]
struct OpenAIResponse {
    data: Vec<OpenAIEmbedding>,
}

#[derive(Deserialize)]
struct OpenAIEmbedding {
    embedding: Vec<f32>,
}

#[derive(Deserialize)]
struct OpenAIError {
    error: OpenAIErrorDetail,
}

#[derive(Deserialize)]
struct OpenAIErrorDetail {
    message: String,
}

impl OpenAIProvider {
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            model,
        }
    }

    fn dimension_for_model(model: &str) -> usize {
        match model {
            "text-embedding-3-large" => OPENAI_LARGE_DIMENSION,
            "text-embedding-3-small" => OPENAI_SMALL_DIMENSION,
            "text-embedding-ada-002" => OPENAI_ADA_DIMENSION,
            _ => OPENAI_LARGE_DIMENSION,
        }
    }
}

#[async_trait]
impl EmbeddingProvider for OpenAIProvider {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let results = self.embed_batch(&[text]).await?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| EmbeddingError::InvalidResponse("Empty response".to_string()))
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let request = OpenAIRequest {
            input: texts.to_vec(),
            model: &self.model,
        };

        let response = self
            .client
            .post(OPENAI_API_URL)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            if response.status() == 429 {
                return Err(EmbeddingError::RateLimited(60));
            }
            let error: OpenAIError = response.json().await.map_err(|e| {
                EmbeddingError::InvalidResponse(format!("Failed to parse error: {}", e))
            })?;
            return Err(EmbeddingError::ApiError(error.error.message));
        }

        let openai_response: OpenAIResponse = response.json().await.map_err(|e| {
            EmbeddingError::InvalidResponse(format!("Failed to parse response: {}", e))
        })?;

        Ok(openai_response
            .data
            .into_iter()
            .map(|e| e.embedding)
            .collect())
    }

    fn dimension(&self) -> usize {
        Self::dimension_for_model(&self.model)
    }
}
