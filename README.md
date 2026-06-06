# ternary-dispatch

Async dispatch of ternary-packed GPU kernels. Queue ordering, conservation verification, and throughput measurement.

## Why This Matters

# ternary-dispatch
Async dispatch of ternary-packed GPU kernels.
Tests queue ordering, ternary conservation after processing, and throughput.

## The Five-Layer Stack

This crate is part of the **Oxide Stack** — a distributed GPU runtime built on five layers:

```
┌─────────────────┐
│  cudaclaw        │  Persistent GPU kernels, warp consensus, SmartCRDT
├─────────────────┤
│  cuda-oxide      │  Flux → MIR → Pliron → NVVM → PTX compiler
├─────────────────┤
│  flux-core       │  Bytecode VM + A2A agent protocol
├─────────────────┤
│  pincher         │  "Vector DB as runtime, LLM as compiler"
├─────────────────┤
│  open-parallel   │  Async runtime (tokio fork)
└─────────────────┘
```

The key insight: **ternary values {-1, 0, +1} map directly to GPU compute**. They pack 16× denser than FP32, enable XNOR+popcount matmul, and conservation laws become compile-time checks.

## Design

Every value in this crate follows **ternary algebra** (Z₃):

| Value | Meaning | GPU Analog |
|-------|---------|------------|
| +1 | Positive / Active / Healthy | Warp vote yes |
| 0 | Neutral / Pending / Balanced | Warp vote abstain |
| -1 | Negative / Failed / Overloaded | Warp vote no |

This isn't arbitrary — ternary is the natural encoding for:
1. **BitNet b1.58** (Microsoft) — ternary LLMs at 60% less power
2. **GPU warp voting** — hardware ballot returns ternary consensus
3. **Conservation laws** — {-1, 0, +1} preserves quantity

## Key Types

```rust
pub struct TritPack
pub fn new
pub fn get
pub fn unpack
pub fn tadd
pub fn sum
pub enum KernelOp
pub struct DispatchResult
pub struct TernaryDispatcher
pub fn new
pub fn enqueue
pub fn queue_depth
```

## Usage

```toml
[dependencies]
ternary-dispatch = "0.1.0"
```

```rust
use ternary_dispatch::*;
// See src/lib.rs tests for complete working examples
```

## Testing

```bash
git clone https://github.com/SuperInstance/ternary-dispatch.git
cd ternary-dispatch
cargo test    # 8 tests
```

## Stats

| Metric | Value |
|--------|-------|
| Tests | 8 |
| Lines of Rust | 230 |
| Public API | 17 items |

## License

Apache-2.0
