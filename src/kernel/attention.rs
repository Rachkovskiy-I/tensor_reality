// =============================================================================
// ATTENTION - Fundamental force between tokens (with parallel computing)
// =============================================================================

use super::token::Token;
use super::HIDDEN_DIM;
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct AttentionMatrix {
    pub scores: Vec<Vec<f32>>,
    pub weights: Vec<Vec<f32>>,
}

impl AttentionMatrix {
    pub fn new() -> Self {
        Self {
            scores: Vec::new(),
            weights: Vec::new(),
        }
    }

    pub fn compute(&mut self, tokens: &[Token]) {
        let size = tokens.len();
        if size == 0 {
            return;
        }

        // Создаем матрицы нужного размера
        self.scores = vec![vec![0.0; size]; size];
        self.weights = vec![vec![0.0; size]; size];

        // Параллельно вычисляем оценки внимания
        let embeddings: Vec<&[f32]> = tokens.iter().map(|t| &t.embedding[..]).collect();
        
        let scores: Vec<Vec<f32>> = (0..size)
            .into_par_iter()
            .map(|i| {
                let mut row = vec![0.0; size];
                let qi = embeddings[i];
                for j in 0..size {
                    let kj = embeddings[j];
                    let score = qi.iter().zip(kj).map(|(a, b)| a * b).sum::<f32>();
                    row[j] = score / (HIDDEN_DIM as f32).sqrt();
                }
                row
            })
            .collect();

        self.scores = scores;

        // Параллельно применяем softmax к каждой строке
        let weights: Vec<Vec<f32>> = self.scores
            .par_iter()
            .map(|row| self.softmax(row))
            .collect();

        self.weights = weights;
    }

    fn softmax(&self, logits: &[f32]) -> Vec<f32> {
        let max = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_sum: f32 = logits.iter().map(|x| (x - max).exp()).sum();
        
        if exp_sum == 0.0 {
            vec![1.0 / logits.len() as f32; logits.len()]
        } else {
            logits.iter().map(|x| (x - max).exp() / exp_sum).collect()
        }
    }

    pub fn entropy(&self) -> f32 {
        let mut total = 0.0;
        let mut count = 0;
        
        for row in &self.weights {
            for &w in row {
                if w > 0.0 {
                    total -= w * w.ln();
                    count += 1;
                }
            }
        }
        
        if count > 0 { total / count as f32 } else { 0.0 }
    }
}