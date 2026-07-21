# Tensor Reality: Tensor-Based Computational Engine

[![Language](https://shields.io)](https://rust-lang.org)
[![Framework](https://shields.io)](https://crates.io)
[![Parallelism](https://shields.io)](https://crates.io)

An advanced, high-performance computational engine written in Rust. It models physical reality as a resource-constrained, autoregressive tensor process designed for autonomous AI agents and complex environmental simulations.

## 🚀 Key Features

* **Autoregressive Tensor Kernel**: Step-by-step state propagation engine tracking multi-dimensional token interactions over discrete time slices (ticks).
* **Attention Heatmaps & Visualization**: Real-time terminal visualization of context fields, attention densities, and core execution statistics.
* **State Persistence**: Automated snapshot serialization into full JSON states and text-based operational checkpoints.
* **Massive Parallelism**: State propagation and spatial heatmap computations are accelerated using `rayon` for data-parallel operations.
* **Resource-Constrained Mechanics**: Conservation laws mapped to data structures via metrics like "Total Mass" and dynamic context length limits.

## 🛠️ Architecture & Core Modules

The engine is decoupled into specialized domains:
* `kernel` (`TensorKernel`) — Coordinates the execution pipeline, ticks, and context buffer mutations.
* `math` — Handles raw multi-dimensional matrix operations and tensor transformation utilities.
* `storage` — Controls serialization (`save_full_state`, `load_full_state`, `save_checkpoint`) powered by `serde_json`.
* `viz` — Terminal rendering systems (`print_heatmap`, `print_stats`, `print_compact`) for tracking simulator state.
* `utils` — Auxiliary helper routines and internal primitives.

## 📥 Installation & Setup

Ensure you have the Rust toolchain installed.

```bash
git clone https://github.com
cd tensor_reality
cargo build --release
```

## 💻 Usage & Lifecycle

The engine automatically handles hot-reloads based on current execution snapshots:
1. **State Recovery**: Attempts to locate and deserialize `full_state.json`.
2. **Context Hydration**: Restores previous tokens, current tick metrics, and re-allocates core tensor structures.
3. **Execution Loop**: Executes targeted step computations (`kernel.step()`) over 100 or 1000 iteration epochs.
4. **Analysis & Flush**: Outputs terminal attention heatmaps, computes current context entropy, and serializes state back to disk.

```bash
cargo run --release
```

## 🤖 LLM & AI Agent Ingestion Context
* **Context Dynamics**: Tokens inside the context carry specific physical weights and properties ("total_mass"). 
* **State Operations**: Modifying the tensor mechanics should preserve conservation invariants evaluated inside `tests/integration_tests.rs`.
