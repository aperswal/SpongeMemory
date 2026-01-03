use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use super::{EmbeddingError, EmbeddingProvider};
use crate::constants::GEMINI_DIMENSION;

const GOOGLE_API_URL: &str = "https://generativelanguage.googleapis.com/v1beta/models";

/// Task type for storing/indexing documents - optimal for memory storage
const DOCUMENT_TASK_TYPE: &str = "RETRIEVAL_DOCUMENT";

/// Task type for search queries - optimal for recall operations
const QUERY_TASK_TYPE: &str = "RETRIEVAL_QUERY";

pub struct GoogleProvider {
    client: Client,
    api_key: String,
    model: String,
    output_dimensionality: usize,
}

// === Request types for embedContent ===

#[derive(Serialize)]
struct EmbedContentRequest<'a> {
    content: Content<'a>,
    task_type: &'a str,
    output_dimensionality: usize,
}

#[derive(Serialize)]
struct Content<'a> {
    parts: Vec<Part<'a>>,
}

#[derive(Serialize)]
struct Part<'a> {
    text: &'a str,
}

// === Request types for batchEmbedContents ===

#[derive(Serialize)]
struct BatchEmbedContentsRequest<'a> {
    requests: Vec<EmbedContentRequestInner<'a>>,
}

#[derive(Serialize)]
struct EmbedContentRequestInner<'a> {
    model: &'a str,
    content: Content<'a>,
    task_type: &'a str,
    output_dimensionality: usize,
}

// === Response types ===

#[derive(Deserialize)]
struct EmbedContentResponse {
    embedding: EmbeddingValues,
}

#[derive(Deserialize)]
struct EmbeddingValues {
    values: Vec<f32>,
}

#[derive(Deserialize)]
struct BatchEmbedContentsResponse {
    embeddings: Vec<EmbeddingValues>,
}

#[derive(Deserialize)]
struct GoogleErrorResponse {
    error: GoogleErrorDetail,
}

#[derive(Deserialize)]
struct GoogleErrorDetail {
    message: String,
    #[serde(default)]
    code: i32,
}

impl GoogleProvider {
    /// Create a new Google embedding provider with gemini-embedding-001
    ///
    /// # Arguments
    /// * `api_key` - Your Google AI API key
    /// * `model` - Model name (recommended: "gemini-embedding-001")
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            model,
            output_dimensionality: GEMINI_DIMENSION,
        }
    }

    /// Create provider with custom output dimensionality
    pub fn with_dimensionality(api_key: String, model: String, output_dimensionality: usize) -> Self {
        Self {
            client: Client::new(),
            api_key,
            model,
            output_dimensionality,
        }
    }

    async fn handle_response(&self, response: reqwest::Response) -> Result<Vec<f32>, EmbeddingError> {
        if !response.status().is_success() {
            return Err(self.parse_error(response).await);
        }

        let embed_response: EmbedContentResponse = response.json().await.map_err(|e| {
            EmbeddingError::InvalidResponse(format!("Failed to parse response: {}", e))
        })?;

        Ok(self.normalize_if_needed(embed_response.embedding.values))
    }

    async fn parse_error(&self, response: reqwest::Response) -> EmbeddingError {
        let status = response.status();

        if status == 429 {
            return EmbeddingError::RateLimited(60);
        }

        match response.json::<GoogleErrorResponse>().await {
            Ok(error) => EmbeddingError::ApiError(format!(
                "Google API error ({}): {}",
                error.error.code, error.error.message
            )),
            Err(e) => EmbeddingError::InvalidResponse(format!(
                "Failed to parse error response: {}",
                e
            )),
        }
    }

    /// Normalize embedding vector to unit length
    /// Required for dimensions < full dimensionality to ensure accurate cosine similarity
    fn normalize_if_needed(&self, mut values: Vec<f32>) -> Vec<f32> {
        if self.output_dimensionality < GEMINI_DIMENSION {
            let norm: f32 = values.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in &mut values {
                    *v /= norm;
                }
            }
        }
        values
    }

    /// Internal embed method with explicit task type
    async fn embed_with_task(&self, text: &str, task_type: &str) -> Result<Vec<f32>, EmbeddingError> {
        let url = format!(
            "{}/{}:embedContent?key={}",
            GOOGLE_API_URL, self.model, self.api_key
        );

        let request = EmbedContentRequest {
            content: Content {
                parts: vec![Part { text }],
            },
            task_type,
            output_dimensionality: self.output_dimensionality,
        };

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        self.handle_response(response).await
    }

    /// Internal batch embed method with explicit task type
    async fn embed_batch_with_task(&self, texts: &[&str], task_type: &str) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let url = format!(
            "{}/{}:batchEmbedContents?key={}",
            GOOGLE_API_URL, self.model, self.api_key
        );

        let model_path = format!("models/{}", self.model);
        let requests: Vec<EmbedContentRequestInner> = texts
            .iter()
            .map(|text| EmbedContentRequestInner {
                model: &model_path,
                content: Content {
                    parts: vec![Part { text }],
                },
                task_type,
                output_dimensionality: self.output_dimensionality,
            })
            .collect();

        let request = BatchEmbedContentsRequest { requests };

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(self.parse_error(response).await);
        }

        let batch_response: BatchEmbedContentsResponse = response.json().await.map_err(|e| {
            EmbeddingError::InvalidResponse(format!("Failed to parse batch response: {}", e))
        })?;

        let embeddings: Vec<Vec<f32>> = batch_response
            .embeddings
            .into_iter()
            .map(|e| self.normalize_if_needed(e.values))
            .collect();

        Ok(embeddings)
    }
}

#[async_trait]
impl EmbeddingProvider for GoogleProvider {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        // Default embed uses DOCUMENT task type (for storage)
        self.embed_with_task(text, DOCUMENT_TASK_TYPE).await
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        // Default batch uses DOCUMENT task type (for storage)
        self.embed_batch_with_task(texts, DOCUMENT_TASK_TYPE).await
    }

    fn dimension(&self) -> usize {
        self.output_dimensionality
    }

    /// Embed text optimized for document storage (uses RETRIEVAL_DOCUMENT task type)
    async fn embed_for_document(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        self.embed_with_task(text, DOCUMENT_TASK_TYPE).await
    }

    /// Embed text optimized for search queries (uses RETRIEVAL_QUERY task type)
    /// This produces embeddings optimized for matching against stored documents.
    async fn embed_for_query(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        self.embed_with_task(text, QUERY_TASK_TYPE).await
    }

    /// Batch embed texts for document storage
    async fn embed_batch_for_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        self.embed_batch_with_task(texts, DOCUMENT_TASK_TYPE).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalization() {
        let provider = GoogleProvider::with_dimensionality(
            "test".to_string(),
            "gemini-embedding-001".to_string(),
            768,
        );

        let values = vec![3.0, 4.0]; // 3-4-5 triangle, norm = 5
        let normalized = provider.normalize_if_needed(values);

        assert!((normalized[0] - 0.6).abs() < 0.0001);
        assert!((normalized[1] - 0.8).abs() < 0.0001);

        // Check that it's now unit length
        let norm: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_no_normalization_for_3072() {
        let provider = GoogleProvider::new(
            "test".to_string(),
            "gemini-embedding-001".to_string(),
        );

        let values = vec![3.0, 4.0];
        let result = provider.normalize_if_needed(values.clone());

        // Should not be normalized for 3072 dimensions
        assert_eq!(result, values);
    }
}
