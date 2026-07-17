// =============================================================================
// KERNEL MODULE - Core computational engine
// =============================================================================

pub mod token;
pub mod context;
pub mod attention;

use context::ContextWindow;
use token::Token;
use attention::AttentionMatrix;

pub const MAX_CONTEXT_LENGTH: usize = 8192;
pub const HIDDEN_DIM: usize = 128;

#[derive(Clone)]
pub struct TensorKernel {
    pub tick: u64,
    pub context: ContextWindow,
    pub attention: AttentionMatrix,
}

impl TensorKernel {
    pub fn new() -> Self {
        println!("[KERNEL] Initializing TensorKernel...");
        Self {
            tick: 0,
            context: ContextWindow::new(),
            attention: AttentionMatrix::new(),
        }
    }

    pub fn step(&mut self) {
        self.tick += 1;
        
        // Добавляем новый токен
        let token = Token::new(self.tick as u32, self.context.len());
        match self.context.push(token) {
            Ok(_) => {
                // Пересчитываем внимание на всех токенах
                self.attention.compute(&self.context.tokens);
                
                if self.tick % 10 == 0 {
                    let entropy = self.attention.entropy();
                    println!("[KERNEL] Tick: {} | Tokens: {} | Mass: {:.2} | Entropy: {:.4}", 
                        self.tick, 
                        self.context.len(),
                        self.context.total_mass(),
                        entropy
                    );
                }
            }
            Err(e) => {
                println!("[KERNEL] Warning: {}", e);
            }
        }
    }

    pub fn context_len(&self) -> usize {
        self.context.len()
    }

    pub fn total_mass(&self) -> f32 {
        self.context.total_mass()
    }

    pub fn attention_entropy(&self) -> f32 {
        self.attention.entropy()
    }
}