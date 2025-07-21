mod card;
mod engine;
mod moves;
mod reward;
mod state;

use pyo3::prelude::*;
use serde_json;

pub use card::Card;
pub use engine::Engine;
pub use moves::Move;
pub use reward::compute_base_reward;
pub use state::GameState;

/// Nombre de cartes dans un jeu
pub const N_CARDS: u32 = 52;
/// Nombre de piles de tableau
pub const N_PILES: u32 = 7;
/// Nombre de fondations
pub const N_FOUNDATIONS: u32 = 4;
/// Nombre de couleurs
pub const N_SUITS: u32 = 4;

#[pyfunction]
pub fn new_game() -> PyResult<String> {
    let state = GameState::new();
    Ok(serde_json::to_string(&state).unwrap())
}

#[pyfunction]
pub fn legal_moves(state: &str) -> PyResult<Vec<String>> {
    let state: GameState = serde_json::from_str(state)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let engine = Engine::from_state(state);
    let moves: Vec<String> = engine
        .get_available_moves()
        .iter()
        .map(|m| serde_json::to_string(m).unwrap())
        .collect();
    Ok(moves)
}

#[pyfunction]
pub fn play_move(state: &str, action: &str) -> PyResult<(String, bool)> {
    let state: GameState = serde_json::from_str(state)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let action: Move = serde_json::from_str(action)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let mut engine = Engine::from_state(state);
    let valid = engine.make_move(&action);
    let new_state = engine.get_state().clone();
    Ok((serde_json::to_string(&new_state).unwrap(), valid))
}

#[pyfunction]
pub fn encode_observation(state: &str) -> PyResult<Vec<f32>> {
    let state: GameState = serde_json::from_str(state)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    Ok(state.encode_observation())
}

#[pyfunction]
pub fn foundation_count(state: &str) -> PyResult<usize> {
    let state: GameState = serde_json::from_str(state)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    Ok(state.foundation_count())
}

#[pyfunction]
pub fn is_won(state: &str) -> PyResult<bool> {
    let state: GameState = serde_json::from_str(state)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    Ok(state.is_won())
}

#[pyfunction]
pub fn compute_base_reward_json(game_state_json: &str) -> PyResult<f32> {
    let state: GameState = serde_json::from_str(game_state_json)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    Ok(reward::compute_base_reward(&state))
}

#[pyfunction]
pub fn move_index(action: &str) -> PyResult<usize> {
    let m: Move = serde_json::from_str(action)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    Ok(m.get_move_index())
}

#[pyfunction]
pub fn move_from_index(index: usize) -> PyResult<Option<String>> {
    Ok(Move::from_move_index(index).map(|m| serde_json::to_string(&m).unwrap()))
}

#[pymodule]
fn klondike_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(new_game, m)?)?;
    m.add_function(wrap_pyfunction!(legal_moves, m)?)?;
    m.add_function(wrap_pyfunction!(play_move, m)?)?;
    m.add_function(wrap_pyfunction!(encode_observation, m)?)?;
    m.add_function(wrap_pyfunction!(foundation_count, m)?)?;
    m.add_function(wrap_pyfunction!(is_won, m)?)?;
    m.add_function(wrap_pyfunction!(compute_base_reward_json, m)?)?;
    m.add_function(wrap_pyfunction!(move_index, m)?)?;
    m.add_function(wrap_pyfunction!(move_from_index, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_game_constants() {
        assert_eq!(N_CARDS, 52);
        assert_eq!(N_PILES, 7);
        assert_eq!(N_FOUNDATIONS, 4);
        assert_eq!(N_SUITS, 4);
    }
}
