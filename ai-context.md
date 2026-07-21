# Tensor Reality — AI Context & Prompting Guide

You are an expert Rust engineer specializing in `ndarray`, parallel computations via `rayon`, and state-machine simulations. Use this guide to reason about the codebase architecture.

## 🛠️ Architecture Guidelines

### 1. The Kernel (`src/kernel.rs`)
* The main state machine is managed by the `TensorKernel` struct.
* It exposes a `.step()` method to advance the system state by one discrete time slice (tick).
* Features `.context_len()` and `.total_mass()` methods to track memory allocation and structural mass conservation invariants.
* Tokens are dynamically stored inside an internal context buffer (`kernel.context`).

### 2. Math Operations (`src/math.rs`)
* Heavy math, tensor transformations, and matrix operations must utilize `ndarray::Array`.
* Implement parallel iterators via `.par_iter()` (`rayon`) for multi-dimensional operations where data size justifies the thread overhead.

### 3. State Management (`src/storage.rs`)
* State is persisted on disk using two distinct mechanisms:
  * `full_state.json`: A complete serialized JSON dump of the kernel state, ticks, and token attributes using `serde_json`.
  * `checkpoint.txt`: A human-readable text file tracking performance benchmarks and operational snapshots.
* When adding fields to `TensorKernel`, ensure they are mirrored in the state structs and properly marked with `#[derive(Serialize, Deserialize)]`.

### 4. Terminal Visualization (`src/viz.rs`)
* Contains pure console UI rendering pipelines.
* `print_heatmap`: Renders attention fields or dimensional layouts onto the console using dense block matrices.
* `print_compact` and `print_stats`: Prints standard telemetry data every N steps (e.g., 20 or 100 ticks) during execution cycles. Never bloat IO operations inside heavy computational loops.

## 🤖 Coding Standards for AI
* **Idiomatic Rust**: Handle errors explicitly using `Result`. Pass errors upstream to `main.rs` instead of invoking `panic!`, `unwrap()`, or `expect()` in core modules.
* **Side Effects**: Use `let _ = ...` intentionally only for uncritical disk flushes or visualization diagnostics.
