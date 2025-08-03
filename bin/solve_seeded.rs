use std::env;
use std::time::Instant;
use core::num::NonZeroU8;
use serde_json::Value;

use lonelybot::shuffler::default_shuffle;
use lonelybot::state::Solitaire;
use lonelybot::solver::{solve_with_tracking, SearchResult};
use lonelybot::tracking::{SearchStatistics, DefaultTerminateSignal};

fn decode_state(data: &str) -> Option<Solitaire> {
    let enc: u64 = data.parse().ok()?;
    Some(Solitaire::from_encode(enc))
}

struct Counter {
    visited: std::cell::Cell<u64>,
    max_depth: std::cell::Cell<usize>,
}

impl Counter {
    fn new() -> Self {
        Self { visited: std::cell::Cell::new(0), max_depth: std::cell::Cell::new(0) }
    }
}

impl SearchStatistics for Counter {
    fn hit_a_state(&self, depth: usize) {
        self.visited.set(self.visited.get() + 1);
        if depth > self.max_depth.get() {
            self.max_depth.set(depth);
        }
    }
    fn hit_unique_state(&self, depth: usize, _n_moves: u32) {
        self.hit_a_state(depth);
    }
    fn finish_move(&self, _depth: usize) {}
}

fn main() {
    let arg = env::args().nth(1).expect("seed or state required");
    let mut game = if let Ok(seed) = arg.parse::<u64>() {
        let cards = default_shuffle(seed);
        Solitaire::new(&cards, NonZeroU8::new(1).unwrap())
    } else {
        let v: Value = serde_json::from_str(&arg).expect("invalid json");
        let encoded = v.get("encoded").and_then(|e| e.as_str()).expect("missing encoded");
        decode_state(encoded).expect("invalid state")
    };

    let counter = Counter::new();
    let start = Instant::now();
    let (status, history) = solve_with_tracking(&mut game, &counter, &DefaultTerminateSignal {});
    let duration = start.elapsed();
    let moves = history.as_ref().map(|h| h.len()).unwrap_or(0);
    let depth = counter.max_depth.get();
    let visited = counter.visited.get();
    let solved = status == SearchResult::Solved;
    let json = serde_json::json!({
        "solved": solved,
        "moves": moves,
        "depth": depth,
        "visited": visited,
        "duration_ms": duration.as_millis() as u64,
    });
    println!("{}", json);
}

