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
    
    // Сохраняем базовую информацию
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

pub fn load_checkpoint(filename: &str) -> Result<String, String> {
    let data = std::fs::read_to_string(filename)
        .map_err(|e| format!("Failed to read checkpoint: {}", e))?;
    Ok(data)
}