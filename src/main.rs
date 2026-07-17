// =============================================================================
// TENSOR_REALITY - Tensor-Based Computational Engine
// =============================================================================

mod kernel;
mod math;
mod utils;
mod storage;

use kernel::TensorKernel;
use storage::{save_checkpoint, load_checkpoint};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║              TENSOR_REALITY v0.1.0                          ║");
    println!("║         Tensor-Based Computational Engine                   ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Пытаемся загрузить сохраненное состояние
    match load_checkpoint("checkpoint.txt") {
        Ok((tick, context_size, total_mass)) => {
            println!("✓ Checkpoint loaded!");
            println!("  Previous tick: {}", tick);
            println!("  Previous context size: {}", context_size);
            println!("  Previous total mass: {:.4}", total_mass);
            println!();
            
            // Создаем ядро и восстанавливаем состояние
            let mut kernel = TensorKernel::new();
            // Восстанавливаем счетчик тиков
            kernel.tick = tick;
            
            println!("Starting main loop from checkpoint...");
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
        }
        Err(_) => {
            println!("No checkpoint found. Starting fresh...");
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
    }
}