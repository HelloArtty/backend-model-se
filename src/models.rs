use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PredictionResult {
    pub predicted_class: i64,
    pub confidence: f32,
}
