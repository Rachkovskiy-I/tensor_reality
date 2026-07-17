// =============================================================================
// TENSOR_REALITY - Tensor-Based Computational Engine
// =============================================================================

mod kernel;
mod math;
mod utils;
mod storage;

use kernel::TensorKernel;
use storage::{save_checkpoint, load_checkpoint, save_full_state, load_full_state};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║              TENSOR_REALITY v0.1.0                          ║");
    println!("║         Tensor-Based Computational Engine                   ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Пытаемся загрузить полное состояние
    match load_full_state("full_state.json") {
        Ok(state) => {
            println!("✓ Full state loaded!");
            println!("  Previous tick: {}", state.tick);
            println!("  Tokens count: {}", state.tokens.len());
            println!();
            
            // Создаем ядро
            let mut kernel = TensorKernel::new();
            kernel.tick = state.tick;
            
            // Восстанавливаем токены
            for token in state.tokens {
                let _ = kernel.context.push(token);
            }
            
            println!("Starting main loop from full state...");
            for _ in 0..100 {
                kernel.step();
            }
            println!();
            println!("✓ Main loop completed");
            println!("  Total ticks: {}", kernel.tick);
            println!("  Final context size: {}", kernel.context_len());
            println!("  Final total mass: {:.4}", kernel.total_mass());
            
            // Сохраняем обновленное состояние
            println!();
            let _ = save_checkpoint(&kernel, "checkpoint.txt");
            let _ = save_full_state(&kernel, "full_state.json");
        }
        Err(_) => {
            println!("No full state found. Starting fresh...");
            println!();
            
            let mut kernel = TensorKernel::new();
            println!("✓ Kernel initialized");
            println!("  Context size: {}", kernel.context_len());
            println!("  Total mass: {:.4}", kernel.total_mass());
            println!();

            println!("Starting main loop...");
            for _ in 0..1000 {
                kernel.step();
            }
            println!();
            println!("✓ Main loop completed");
            println!("  Total ticks: {}", kernel.tick);
            println!("  Final context size: {}", kernel.context_len());
            println!("  Final total mass: {:.4}", kernel.total_mass());

            // Сохраняем состояние
            println!();
            let _ = save_checkpoint(&kernel, "checkpoint.txt");
            let _ = save_full_state(&kernel, "full_state.json");
        }
    }
}