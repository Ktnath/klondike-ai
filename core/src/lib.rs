#![allow(dead_code)]
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

/// Create a new game and return its JSON representation.
#[pyfunction]
pub fn new_game(seed: Option<&str>) -> PyResult<String> {
    let _ = seed; // seed currently unused in minimal implementation
    let state = serde_json::json!({
        "stock": [],
        "waste": [],
        "tableau": [[], [], [], [], [], [], []],
        "foundations": [[], [], [], []],
        "seed": seed
    });
    Ok(state.to_string())
}

/// Return the list of legal moves for a given JSON state.
#[pyfunction]
pub fn legal_moves(_state: &str) -> PyResult<Vec<String>> {
    Ok(Vec::new())
}

/// Apply a move to the given JSON state. Returns the new state and a validity flag.
#[pyfunction]
pub fn play_move(state: &str, _mv: &str) -> PyResult<(String, bool)> {
    Ok((state.to_string(), false))
}

/// Encode a game state into a feature vector (placeholder).
#[pyfunction]
pub fn encode_observation(_state: &str) -> PyResult<Vec<f32>> {
    Ok(Vec::new())
}

/// Count how many cards are already in the foundations.
#[pyfunction]
pub fn foundation_count(state: &str) -> PyResult<usize> {
    let v: serde_json::Value = serde_json::from_str(state)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let cnt = v
        .get("foundations")
        .and_then(|f| f.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|col| col.as_array())
                .map(|c| c.len())
                .sum()
        })
        .unwrap_or(0);
    Ok(cnt)
}

/// Check if the game is won.
#[pyfunction]
pub fn is_won(state: &str) -> PyResult<bool> {
    Ok(foundation_count(state)? >= 52)
}

/// Convert a move JSON into its action index (placeholder).
#[pyfunction]
pub fn move_index(_mv: &str) -> PyResult<usize> {
    Ok(0)
}

/// Convert an action index back to a move JSON (placeholder).
#[pyfunction]
pub fn move_from_index(_idx: usize) -> PyResult<Option<String>> {
    Ok(None)
}
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
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let mut rng = rand::rngs::SmallRng::seed_from_u64(nanos);
    #[allow(deprecated)]
    Ok(rng.gen::<u32>())
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
    // Game management functions
    m.add_function(wrap_pyfunction!(new_game, m)?)?;
    m.add_function(wrap_pyfunction!(legal_moves, m)?)?;
    m.add_function(wrap_pyfunction!(play_move, m)?)?;

    // Observation helpers
    m.add_function(wrap_pyfunction!(encode_observation, m)?)?;
    m.add_function(wrap_pyfunction!(foundation_count, m)?)?;
    m.add_function(wrap_pyfunction!(is_won, m)?)?;

    // Move <-> index conversions
    m.add_function(wrap_pyfunction!(move_index, m)?)?;
    m.add_function(wrap_pyfunction!(move_from_index, m)?)?;

    m.add_function(wrap_pyfunction!(solve_klondike, m)?)?;
    m.add_function(wrap_pyfunction!(shuffle_seed, m)?)?;
    m.add_function(wrap_pyfunction!(compute_base_reward_json, m)?)?;
    Ok(())
}
