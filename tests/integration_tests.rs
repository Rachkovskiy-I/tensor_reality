// =============================================================================
// INTEGRATION TESTS
// =============================================================================

use tensor_reality::kernel::TensorKernel;
use tensor_reality::kernel::token::Token;
use tensor_reality::kernel::context::ContextWindow;
use tensor_reality::kernel::attention::AttentionMatrix;

#[test]
fn test_token_creation() {
    let token = Token::new(1, 0);
    assert_eq!(token.id, 1);
    assert_eq!(token.position, 0);
    assert_eq!(token.embedding.len(), 128);
}

#[test]
fn test_token_mass() {
    let token = Token::new(1, 0);
    let mass = token.compute_mass();
    assert!(mass >= 0.0);
}

#[test]
fn test_context_push() {
    let mut context = ContextWindow::new();
    let token = Token::new(1, 0);
    
    let result = context.push(token);
    assert!(result.is_ok());
    assert_eq!(context.len(), 1);
}

#[test]
fn test_context_overflow() {
    let mut context = ContextWindow::new();
    
    for i in 0..10000 {
        let token = Token::new(i as u32, i);
        let _ = context.push(token);
    }
    
    assert!(context.len() <= 8192);
}

#[test]
fn test_kernel_creation() {
    let kernel = TensorKernel::new();
    assert_eq!(kernel.tick, 0);
    assert_eq!(kernel.context_len(), 0);
}

#[test]
fn test_kernel_step() {
    let mut kernel = TensorKernel::new();
    kernel.step();
    assert_eq!(kernel.tick, 1);
    assert_eq!(kernel.context_len(), 1);
}

#[test]
fn test_attention_compute() {
    let mut attention = AttentionMatrix::new();
    let token1 = Token::new(1, 0);
    let token2 = Token::new(2, 1);
    let tokens = vec![token1, token2];
    
    attention.compute(&tokens);
    assert_eq!(attention.scores.len(), 2);
    assert_eq!(attention.weights.len(), 2);
}

#[test]
fn test_attention_entropy() {
    let mut attention = AttentionMatrix::new();
    let token1 = Token::new(1, 0);
    let token2 = Token::new(2, 1);
    let tokens = vec![token1, token2];
    
    attention.compute(&tokens);
    let entropy = attention.entropy();
    assert!(entropy >= 0.0);
}