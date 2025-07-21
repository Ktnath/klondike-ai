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

#[pymodule]
fn klondike_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_klondike, m)?)?;
    Ok(())
}
