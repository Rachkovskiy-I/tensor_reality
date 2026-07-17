// =============================================================================
// TRAIN - Learning and adaptation
// =============================================================================

use super::token::Token;
use super::context::ContextWindow;
use super::HIDDEN_DIM;

#[derive(Clone, Debug)]
pub struct Trainer {
    pub learning_rate: f32,
    pub epochs: usize,
}

impl Trainer {
    pub fn new() -> Self {
        Self {
            learning_rate: 0.01,
            epochs: 10,
        }
    }

    // Простое обучение: немного изменяем эмбеддинги токенов
    pub fn train_step(&self, context: &mut ContextWindow) {
        if context.len() < 2 {
            return;
        }

        // Простая эвристика: сдвигаем эмбеддинги ближе к среднему
        let avg_embedding = self.compute_average_embedding(context);
        
        for i in 0..context.len() {
            if let Some(token) = context.tokens.get_mut(i) {
                // Обновляем эмбеддинг в сторону среднего
                for j in 0..token.embedding.len().min(avg_embedding.len()) {
                    token.embedding[j] += self.learning_rate * (avg_embedding[j] - token.embedding[j]);
                }
            }
        }
    }

    fn compute_average_embedding(&self, context: &ContextWindow) -> Vec<f32> {
        let mut avg = vec![0.0; HIDDEN_DIM];
        let count = context.len();
        
        for token in &context.tokens {
            for i in 0..HIDDEN_DIM.min(token.embedding.len()) {
                avg[i] += token.embedding[i] / count as f32;
            }
        }
        
        avg
    }

    // Вычисляем "ошибку" как расстояние между токенами
    pub fn compute_loss(&self, context: &ContextWindow) -> f32 {
        if context.len() < 2 {
            return 0.0;
        }

        let mut total_loss = 0.0;
        let mut pairs = 0;

        for i in 0..context.len() {
            for j in i+1..context.len() {
                if let (Some(a), Some(b)) = (context.get(i), context.get(j)) {
                    let dist = self.cosine_distance(&a.embedding, &b.embedding);
                    total_loss += dist;
                    pairs += 1;
                }
            }
        }

        if pairs > 0 {
            total_loss / pairs as f32
        } else {
            0.0
        }
    }

    fn cosine_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            return 1.0;
        }
        1.0 - (dot / (norm_a * norm_b))
    }
}