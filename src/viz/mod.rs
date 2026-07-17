// =============================================================================
// VIZ - Visualization utilities
// =============================================================================

use crate::kernel::TensorKernel;

pub fn print_heatmap(kernel: &TensorKernel) {
    let size = kernel.context_len();
    if size == 0 || size > 50 {
        println!("[VIZ] Too many tokens ({}), skipping heatmap", size);
        return;
    }

    println!("\n[VIZ] Attention Heatmap ({} tokens):", size);
    println!("    {}", "─".repeat(size * 2 + 2));

    for i in 0..size {
        print!("  ");
        for j in 0..size {
            let w = kernel.attention.weights[i][j];
            let symbol = weight_to_symbol(w);
            print!("{}", symbol);
        }
        println!();
    }
    println!("    {}", "─".repeat(size * 2 + 2));
    println!("  ██ = high attention  ·· = low attention\n");
}

fn weight_to_symbol(w: f32) -> &'static str {
    if w > 0.3 {
        "██"
    } else if w > 0.1 {
        "▓▓"
    } else if w > 0.05 {
        "▒▒"
    } else if w > 0.01 {
        "░░"
    } else {
        "··"
    }
}

pub fn print_stats(kernel: &TensorKernel) {
    println!("[VIZ] STATS:");
    println!("  Ticks: {}", kernel.tick);
    println!("  Tokens: {}", kernel.context_len());
    println!("  Total mass: {:.4}", kernel.total_mass());
    println!("  Entropy: {:.4}", kernel.attention_entropy());
    println!("  Overflow count: {}", kernel.context.overflow_count);
}

pub fn print_compact(kernel: &TensorKernel) {
    let tokens = kernel.context_len();
    let mass = kernel.total_mass();
    let entropy = kernel.attention_entropy();
    let overflow = kernel.context.overflow_count;
    
    print!("\r[VIZ] Tokens: {:4} | Mass: {:8.2} | Entropy: {:.4} | Overflow: {:4}",
        tokens, mass, entropy, overflow
    );
}