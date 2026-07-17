// =============================================================================
// STORAGE - Save and load system state
// =============================================================================

use std::fs::File;
use std::io::Write;
use std::path::Path;

use crate::kernel::TensorKernel;

pub fn save_checkpoint(kernel: &TensorKernel, filename: &str) -> Result<(), String> {
    let path = Path::new(filename);
    let mut file = File::create(path).map_err(|e| format!("Failed to create file: {}", e))?;
    
    let data = format!(
        "tick:{}\ncontext_size:{}\ntotal_mass:{:.4}\n",
        kernel.tick,
        kernel.context_len(),
        kernel.total_mass()
    );
    
    file.write_all(data.as_bytes())
        .map_err(|e| format!("Failed to write: {}", e))?;
    
    println!("[STORAGE] Checkpoint saved to: {}", filename);
    Ok(())
}

pub fn load_checkpoint(filename: &str) -> Result<(u64, usize, f32), String> {
    let data = std::fs::read_to_string(filename)
        .map_err(|e| format!("Failed to read checkpoint: {}", e))?;
    
    let mut tick = 0;
    let mut context_size = 0;
    let mut total_mass = 0.0;
    
    for line in data.lines() {
        if let Some(stripped) = line.strip_prefix("tick:") {
            tick = stripped.parse().map_err(|e| format!("Invalid tick: {}", e))?;
        } else if let Some(stripped) = line.strip_prefix("context_size:") {
            context_size = stripped.parse().map_err(|e| format!("Invalid context_size: {}", e))?;
        } else if let Some(stripped) = line.strip_prefix("total_mass:") {
            total_mass = stripped.parse().map_err(|e| format!("Invalid total_mass: {}", e))?;
        }
    }
    
    Ok((tick, context_size, total_mass))
}