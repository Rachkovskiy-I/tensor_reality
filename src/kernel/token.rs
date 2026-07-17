// =============================================================================
// TOKEN - Fundamental particle
// =============================================================================

use super::HIDDEN_DIM;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Token {
    pub id: u32,
    pub embedding: Vec<f32>,
    pub position: usize,
}

impl Token {
    pub fn new(id: u32, position: usize) -> Self {
        let embedding = (0..HIDDEN_DIM)
            .map(|_| rand::random::<f32>() * 2.0 - 1.0)
            .collect();
        
        Self {
            id,
            embedding,
            position,
        }
    }

    pub fn compute_mass(&self) -> f32 {
        self.embedding.iter().map(|x| x * x).sum::<f32>().sqrt()
    }
}