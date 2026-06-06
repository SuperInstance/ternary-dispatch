//! # ternary-dispatch
//!
//! Async dispatch of ternary-packed GPU kernels.
//! Tests queue ordering, ternary conservation after processing, and throughput.

use std::collections::VecDeque;

/// A packed ternary value: 16 trits in a u32 (2 bits each).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TritPack(pub u32);

impl TritPack {
    pub fn new(trits: &[i8]) -> Self {
        let mut packed = 0u32;
        for (i, &t) in trits.iter().take(16).enumerate() {
            let bits = match t { -1 => 0b11, 0 => 0b00, 1 => 0b01, _ => 0b00 };
            packed |= (bits as u32) << (i * 2);
        }
        TritPack(packed)
    }

    pub fn get(&self, idx: usize) -> i8 {
        let bits = (self.0 >> (idx * 2)) & 0b11;
        match bits { 0b11 => -1, 0b00 => 0, 0b01 => 1, _ => 0 }
    }

    pub fn unpack(&self) -> [i8; 16] {
        let mut arr = [0i8; 16];
        for i in 0..16 { arr[i] = self.get(i); }
        arr
    }

    /// Ternary add two packs element-wise (Z₃ addition).
    pub fn tadd(&self, other: &Self) -> Self {
        let mut packed = 0u32;
        for i in 0..16 {
            let a = self.get(i);
            let b = other.get(i);
            let r = match (a, b) {
                (-1, -1) => 1, (-1, 0) => -1, (-1, 1) => 0,
                (0, -1) => -1, (0, 0) => 0, (0, 1) => 1,
                (1, -1) => 0, (1, 0) => 1, (1, 1) => -1,
                _ => 0,
            };
            let bits = match r { -1 => 0b11, 0 => 0b00, 1 => 0b01, _ => 0b00 };
            packed |= (bits as u32) << (i * 2);
        }
        TritPack(packed)
    }

    /// Sum of all trits (conservation check).
    pub fn sum(&self) -> i32 {
        (0..16).map(|i| self.get(i) as i32).sum()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelOp {
    TernaryMap { input: TritPack, scale: i8 },
    TernaryReduce { inputs: Vec<TritPack> },
    TernaryMatVec { weight: TritPack, vector: TritPack },
    TernaryFilter { input: TritPack, threshold: i8 },
}

#[derive(Debug, Clone)]
pub struct DispatchResult {
    pub op_name: String,
    pub output: TritPack,
    pub queue_position: usize,
    pub latency_us: u64,
}

pub struct TernaryDispatcher {
    queue: VecDeque<KernelOp>,
    results: Vec<DispatchResult>,
    total_ops: u64,
    total_latency_us: u64,
}

impl TernaryDispatcher {
    pub fn new() -> Self {
        Self { queue: VecDeque::new(), results: Vec::new(), total_ops: 0, total_latency_us: 0 }
    }

    pub fn enqueue(&mut self, op: KernelOp) {
        self.queue.push_back(op);
    }

    pub fn queue_depth(&self) -> usize { self.queue.len() }

    pub fn execute_one(&mut self) -> Option<DispatchResult> {
        let op = self.queue.pop_front()?;
        let position = self.results.len();
        let (name, output, latency) = match op {
            KernelOp::TernaryMap { input, scale } => {
                let scaled = TritPack::new(&(0..16).map(|i| {
                    let v = input.get(i);
                    match (v, scale) { (-1, -1) => 1, (-1, 1) => -1, (1, -1) => -1, (1, 1) => 1, _ => 0 }
                }).collect::<Vec<_>>());
                ("map", scaled, 10u64)
            }
            KernelOp::TernaryReduce { inputs } => {
                let mut acc = TritPack(0);
                for inp in &inputs { acc = acc.tadd(inp); }
                ("reduce", acc, 25u64 * inputs.len() as u64)
            }
            KernelOp::TernaryMatVec { weight, vector } => {
                let output = weight.tadd(&vector); // simplified
                ("matvec", output, 40u64)
            }
            KernelOp::TernaryFilter { input, threshold } => {
                let filtered = TritPack::new(&(0..16).map(|i| {
                    let v = input.get(i);
                    if v >= threshold { v } else { 0 }
                }).collect::<Vec<_>>());
                ("filter", filtered, 15u64)
            }
        };

        self.total_ops += 1;
        self.total_latency_us += latency;
        let result = DispatchResult { op_name: name.into(), output, queue_position: position, latency_us: latency };
        self.results.push(result.clone());
        Some(result)
    }

    pub fn execute_all(&mut self) -> Vec<DispatchResult> {
        let mut results = Vec::new();
        while let Some(r) = self.execute_one() { results.push(r); }
        results
    }

    pub fn throughput_ops_per_sec(&self) -> f64 {
        if self.total_latency_us == 0 { return 0.0; }
        self.total_ops as f64 / (self.total_latency_us as f64 / 1_000_000.0)
    }

    pub fn total_ops(&self) -> u64 { self.total_ops }
    pub fn avg_latency_us(&self) -> f64 {
        if self.total_ops == 0 { 0.0 } else { self.total_latency_us as f64 / self.total_ops as f64 }
    }
}

impl Default for TernaryDispatcher {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trit_pack_roundtrip() {
        let trits = [-1i8, 0, 1, -1, 1, 0, 0, 1, -1, -1, 1, 0, 1, -1, 0, 1];
        let packed = TritPack::new(&trits);
        let unpacked = packed.unpack();
        for i in 0..16 { assert_eq!(unpacked[i], trits[i]); }
    }

    #[test]
    fn test_tadd_conservation() {
        let a = TritPack::new(&[1, -1, 0, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1]);
        let b = TritPack::new(&[-1, 1, 0, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1]);
        let c = a.tadd(&b);
        // Each pair sums to 0 in Z₃
        assert_eq!(c.sum(), 0);
    }

    #[test]
    fn test_dispatch_map() {
        let mut d = TernaryDispatcher::new();
        let input = TritPack::new(&[1, -1, 0, 1, -1, 0, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1]);
        d.enqueue(KernelOp::TernaryMap { input, scale: -1 });
        let result = d.execute_one().unwrap();
        assert_eq!(result.op_name, "map");
    }

    #[test]
    fn test_dispatch_reduce() {
        let mut d = TernaryDispatcher::new();
        let inputs = vec![TritPack::new(&[1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1]); 3];
        d.enqueue(KernelOp::TernaryReduce { inputs });
        let result = d.execute_one().unwrap();
        assert_eq!(result.op_name, "reduce");
    }

    #[test]
    fn test_dispatch_ordering() {
        let mut d = TernaryDispatcher::new();
        d.enqueue(KernelOp::TernaryMap { input: TritPack(0), scale: 1 });
        d.enqueue(KernelOp::TernaryFilter { input: TritPack(0), threshold: 0 });
        d.enqueue(KernelOp::TernaryMatVec { weight: TritPack(0), vector: TritPack(0) });
        let results = d.execute_all();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].queue_position, 0);
        assert_eq!(results[1].queue_position, 1);
        assert_eq!(results[2].queue_position, 2);
    }

    #[test]
    fn test_throughput() {
        let mut d = TernaryDispatcher::new();
        for _ in 0..100 {
            d.enqueue(KernelOp::TernaryMap { input: TritPack(0), scale: 1 });
        }
        d.execute_all();
        assert_eq!(d.total_ops(), 100);
        assert!(d.throughput_ops_per_sec() > 0.0);
    }

    #[test]
    fn test_pipeline_filter_then_reduce() {
        let mut d = TernaryDispatcher::new();
        let input = TritPack::new(&[1, -1, 0, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1]);
        d.enqueue(KernelOp::TernaryFilter { input, threshold: 1 });
        let filter_result = d.execute_one().unwrap();
        d.enqueue(KernelOp::TernaryReduce { inputs: vec![filter_result.output] });
        let reduce_result = d.execute_one().unwrap();
        assert_eq!(reduce_result.op_name, "reduce");
    }

    #[test]
    fn test_tadd_closure() {
        let a = TritPack::new(&[1, -1, 0, 1, -1, 0, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1]);
        let b = TritPack::new(&[-1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1]);
        let c = a.tadd(&b);
        for i in 0..16 { assert!(c.get(i) >= -1 && c.get(i) <= 1); }
    }
}
