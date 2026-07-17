// =============================================================================
// CONTEXT WINDOW - Container for tokens
// =============================================================================

use super::token::Token;
use super::MAX_CONTEXT_LENGTH;

#[derive(Clone, Debug)]
pub struct ContextWindow {
    pub tokens: Vec<Token>,
    pub max_length: usize,
    pub overflow_count: usize,
}

impl ContextWindow {
    pub fn new() -> Self {
        Self {
            tokens: Vec::with_capacity(MAX_CONTEXT_LENGTH),
            max_length: MAX_CONTEXT_LENGTH,
            overflow_count: 0,
        }
    }

    pub fn push(&mut self, token: Token) -> Result<(), String> {
        if self.tokens.len() >= self.max_length {
            // Удаляем самый старый токен
            let removed = self.tokens.remove(0);
            self.overflow_count += 1;
            self.tokens.push(token);
            return Ok(());
        }
        self.tokens.push(token);
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    pub fn get(&self, position: usize) -> Option<&Token> {
        self.tokens.get(position)
    }

    pub fn total_mass(&self) -> f32 {
        self.tokens.iter().map(|t| t.compute_mass()).sum()
    }
}