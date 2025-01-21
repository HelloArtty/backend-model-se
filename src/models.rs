use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FoodName {
    pub th: String,
    pub en: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FoodItem {
    pub id: i64,
    pub name: FoodName,
}

#[derive(Debug, Serialize)]
pub struct PredictionResponse {
    pub predicted_class: i64,
    pub data: FoodItem,
}
