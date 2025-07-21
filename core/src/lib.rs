use pyo3::prelude::*;
use serde_json;

mod card;
mod convert;
mod deck;
mod engine;
mod formatter;
mod graph;
mod hidden;
mod hop_solver;
mod mcts_solver;
mod moves;
mod pruning;
mod shuffler;
mod solver;
mod stack;
mod standard;
mod state;
mod tracking;
mod traverse;
mod dependencies;
mod partial;
mod analysis;
mod game_theory;
mod utils;

/// Solve a Klondike game represented by a JSON seed.
/// The JSON should be an integer seed used for shuffling.
#[pyfunction]
pub fn solve_klondike(json: &str) -> PyResult<String> {
    let seed: u64 = serde_json::from_str(json)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    use std::num::NonZeroU8;
    let cards = shuffler::default_shuffle(seed);
    let mut game = state::Solitaire::new(&cards, NonZeroU8::new(1).unwrap());
    let (res, history) = solver::solve(&mut game);
    let moves: Vec<String> = history
        .unwrap_or_default()
        .iter()
        .map(|m| format!("{:?}", m))
        .collect();
    let result = serde_json::json!({
        "result": format!("{:?}", res),
        "moves": moves
    });
    Ok(result.to_string())
}

/// Generate a random shuffle seed as a 32-bit integer.
#[pyfunction]
pub fn shuffle_seed() -> PyResult<u32> {
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::SmallRng::from_entropy();
    Ok(rng.gen())
}

/// Compute a sparse reward from a JSON representation of the game state.
///
/// The JSON must contain a `foundations` field that stores four arrays of
/// cards. When all four foundations contain 13 cards the reward is `1.0`,
/// otherwise `0.0`.
#[pyfunction]
pub fn compute_base_reward_json(state: &str) -> PyResult<f32> {
    let v: serde_json::Value = serde_json::from_str(state)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let found_ok = v
        .get("foundations")
        .and_then(|f| f.as_array())
        .map(|arr| arr.iter().all(|col| col.as_array().map(|c| c.len() == 13).unwrap_or(false)))
        .unwrap_or(false);
    Ok(if found_ok { 1.0 } else { 0.0 })
}

#[pymodule]
fn klondike_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_klondike, m)?)?;
    m.add_function(wrap_pyfunction!(shuffle_seed, m)?)?;
    m.add_function(wrap_pyfunction!(compute_base_reward_json, m)?)?;
    Ok(())
}
