# ternary-dispatch

**Async dispatch of ternary-packed GPU kernels. Queue ordering, conservation verification, and throughput measurement — the glue between "here's a ternary operation" and "here's the result, in order, with proof that nothing was corrupted."**

## Why This Exists

GPU kernel dispatch sounds simple: put work on a queue, execute it, return results. But in ternary systems, there's a constraint that doesn't exist in float computation: **conservation**. If you pack 16 trits into a u32, perform operations on them, and unpack the result, the sum of all trits should be predictable. Ternary addition over Z₃ has exact algebraic properties — there's no rounding to hide corruption. If a bit flips during dispatch, you'll see it in the sum.

This crate provides the dispatch infrastructure for ternary-packed GPU kernels with three guarantees:
1. **Ordering** — kernels execute in FIFO order; results come back in the same order
2. **Conservation** — the `TritPack::sum()` method lets you verify that Z₃ arithmetic wasn't corrupted
3. **Throughput measurement** — every operation has a latency, and aggregate stats are tracked automatically

## The Key Insight

A `TritPack` stores 16 ternary values {-1, 0, +1} in a single `u32` — 2 bits per trit:

```
u32:  [t15|t14|t13|t12|t11|t10|t9|t8|t7|t6|t5|t4|t3|t2|t1|t0]
      Each t is 2 bits:  00 = 0, 01 = +1, 11 = -1
```

This is the natural packing for ternary data on binary hardware. 16 trits per u32 means:
- A 1024-element ternary vector fits in 64 u32s (256 bytes)
- Z₃ addition (`tadd`) operates on all 16 trits simultaneously
- GPU kernels can process the packed representation directly

The conservation check is simple: `sum()` adds all 16 trits. For two TritPacks `a` and `b`, the sum of `a.tadd(&b)` should equal `sum(a) + sum(b)` mod 3 (in Z₃ arithmetic). If it doesn't, something went wrong in the pipeline.

## Quick Start

```rust
use ternary_dispatch::*;

// Pack 16 trits into a u32
let trits = [-1i8, 0, 1, -1, 1, 0, 0, 1, -1, -1, 1, 0, 1, -1, 0, 1];
let packed = TritPack::new(&trits);

// Round-trip: unpack and verify
let unpacked = packed.unpack();
assert_eq!(unpacked[0], -1);
assert_eq!(unpacked[1], 0);
assert_eq!(unpacked[2], 1);

// Z₃ addition: a + b where each element wraps mod 3
let a = TritPack::new(&[1, -1, 0, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1]);
let b = TritPack::new(&[-1, 1, 0, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1]);
let c = a.tadd(&b);
assert_eq!(c.sum(), 0);  // Each pair cancels: a[i] + b[i] = 0 in Z₃

// Dispatch kernels through a queue
let mut dispatcher = TernaryDispatcher::new();
dispatcher.enqueue(KernelOp::TernaryMap { input: packed, scale: -1 });
dispatcher.enqueue(KernelOp::TernaryFilter { input: packed, threshold: 1 });
dispatcher.enqueue(KernelOp::TernaryMatVec { weight: a, vector: b });

// Execute in order
let results = dispatcher.execute_all();
assert_eq!(results.len(), 3);
assert_eq!(results[0].queue_position, 0);  // FIFO ordering preserved

// Throughput stats
println!("Ops/sec: {:.0}", dispatcher.throughput_ops_per_sec());
println!("Avg latency: {:.1} µs", dispatcher.avg_latency_us());
```

## Architecture

### TritPack: 16 Trits in a u32

```
Bit layout (LSB first):
  Bits [1:0]   = trit 0    (00=0, 01=+1, 11=-1)
  Bits [3:2]   = trit 1
  Bits [5:4]   = trit 2
  ...
  Bits [31:30] = trit 15

Encoding:  00 → 0 (neutral)
           01 → +1 (positive)
           11 → -1 (negative, 0b11 chosen so sign bit is set)
           10 → unused
```

The `10` encoding is unused. If you see it, the TritPack was corrupted. This gives you a built-in integrity check — in theory, every unpacked trit should be {-1, 0, +1}. If it's something else, you know a bit flipped.

### Kernel Operations

| Operation | Inputs | Description |
|-----------|--------|-------------|
| `TernaryMap` | input + scale | Scale each trit by {-1, 0, +1} (element-wise sign multiply) |
| `TernaryReduce` | Vec of TritPacks | Z₃ addition across all packs |
| `TernaryMatVec` | weight + vector | Ternary matrix-vector multiply (simplified as tadd) |
| `TernaryFilter` | input + threshold | Zero out trits below threshold |

### DispatchResult

```rust
pub struct DispatchResult {
    pub op_name: String,        // "map", "reduce", "matvec", "filter"
    pub output: TritPack,       // Result of the operation
    pub queue_position: usize,  // Position in the original queue (ordering proof)
    pub latency_us: u64,        // Simulated latency in microseconds
}
```

### TernaryDispatcher

| Method | Description |
|--------|-------------|
| `new()` | Create empty dispatcher |
| `enqueue(op)` | Add kernel to the FIFO queue |
| `queue_depth()` | Pending kernel count |
| `execute_one()` | Pop and execute one kernel |
| `execute_all()` | Execute entire queue |
| `throughput_ops_per_sec()` | Aggregate throughput |
| `total_ops()` | Operations completed |
| `avg_latency_us()` | Average latency per operation |

## Conservation Verification

The key safety property of ternary dispatch is that Z₃ arithmetic is *exact*. There's no rounding, no precision loss, no accumulation error. This means you can verify correctness with a simple sum check:

```rust
// Before dispatch
let input_sum = input.sum();

// After dispatch
let output_sum = result.output.sum();

// For a map with scale=1: sums should be equal
// For a reduce: output sum should equal sum of input sums (mod 3)
// Any deviation means something went wrong
```

This is a property that float dispatch can't provide. Float sums accumulate rounding error, and after enough operations, you can't tell if a result is correct or corrupted. In Z₃, the sum is exact or something is broken.

## Real-World Example: Pipeline Processing

```rust
use ternary_dispatch::*;

let mut d = TernaryDispatcher::new();

// Stage 1: Filter out negative trits
let raw = TritPack::new(&[1, -1, 0, 1, -1, 0, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1]);
d.enqueue(KernelOp::TernaryFilter { input: raw, threshold: 1 });

let filter_result = d.execute_one().unwrap();
// filter_result.output has only {0, +1} — negatives zeroed

// Stage 2: Reduce with another pack
let other = TritPack::new(&[0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]);
d.enqueue(KernelOp::TernaryReduce { inputs: vec![filter_result.output, other] });

let reduce_result = d.execute_one().unwrap();

// Stage 3: Scale by -1 (invert)
d.enqueue(KernelOp::TernaryMap { input: reduce_result.output, scale: -1 });

let final_result = d.execute_one().unwrap();
println!("Final: {:?}", final_result.output.unpack());

// Throughput
println!("Pipeline: 3 ops, {:.1} µs avg", d.avg_latency_us());
```

## Design Decisions

**Simulated latency** — The latency values (10-40 µs) are simulated, not measured. In a real GPU dispatch system, latency depends on the hardware. The simulated values let you test pipeline logic without a GPU. Replace them with real measurements in production.

**FIFO ordering** — Strict first-in-first-out. No priority queue, no speculative execution, no out-of-order completion. This keeps the ordering guarantee simple and verifiable.

**u32 packing** — 16 trits per u32 leaves 0 bits unused (32 bits / 2 bits per trit = 16 trits). No wasted space. The packing could be extended to u64 or u128 for SIMD operations.

**No async runtime** — Despite the name "async dispatch," there's no tokio/futures dependency. The dispatch is synchronous and deterministic. "Async" refers to the queued execution model, not the Rust async keyword.

## Ecosystem Connections

- **`ternary-fuse`** — Fused kernels that the dispatcher executes
- **`ternary-interpreter`** — VM for ternary control flow (dispatch decisions)
- **`ternary-matmul`** — Matrix multiply kernels
- **`ternary-kernel-launch`** — Lower-level GPU kernel launch infrastructure
- **`ternary-accumulator`** — Gradient accumulation (training counterpart)

## Open Questions

- **Real GPU latencies**: The simulated latencies are placeholders. Real GPU dispatch needs measured latencies from CUDA/HIP kernels.
- **Batching**: Should the dispatcher support batch submission? (Enqueue multiple ops, execute as a single batch.)
- **Error handling**: Currently operations can't fail. Real GPU dispatch needs error codes, retries, and timeout handling.
- **Out-of-order execution**: FIFO is simple but suboptimal for independent kernels. Could independent ops be executed in parallel?

## Stats

| Metric | Value |
|--------|-------|
| Lines of Rust | ~230 |
| Tests | 8 |
| Public API | 17 items |
| Dependencies | 0 |

## License

Apache-2.0
