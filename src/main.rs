// =============================================================================
// TENSOR_REALITY - Tensor-Based Computational Engine
// =============================================================================

mod kernel;
mod math;
mod utils;
mod storage;

use kernel::TensorKernel;
use storage::save_checkpoint;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║              TENSOR_REALITY v0.1.0                          ║");
    println!("║         Tensor-Based Computational Engine                   ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
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
}