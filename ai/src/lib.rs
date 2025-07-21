mod network;
mod mcts;
mod coach;

use pyo3::prelude::*;
use serde_json;
/// Placeholder types used for compiling the examples.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct GameState {}

impl GameState {
    pub fn encode_observation(&self) -> Vec<f32> { Vec::new() }
    pub fn is_won(&self) -> bool { false }
    pub fn get_score(&self) -> i32 { 0 }
}

#[derive(Debug, Clone)]
pub struct Engine {}

impl Engine {
    pub fn new() -> Self { Self {} }
    pub fn from_state(_state: GameState) -> Self { Self {} }
    pub fn get_state(&self) -> GameState { GameState {} }
    pub fn get_available_moves(&self) -> Vec<Move> { Vec::new() }
    pub fn make_move(&mut self, _m: &Move) {}
}

#[derive(Debug, Clone, Copy)]
pub struct Move;

impl Move {
    pub fn get_move_index(&self) -> usize { 0 }
}

pub use network::NeuralNet;
pub use mcts::MCTS;
pub use coach::Coach;

#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub num_iterations: i32,
    pub num_episodes: i32,
    pub temp_threshold: i32,
    pub update_threshold: f32,
    pub max_moves: i32,
    pub num_mcts_sims: i32,
    pub arena_compare: i32,
    pub cpuct: f32,
    pub checkpoint_interval: i32,
    pub load_model: bool,
    pub train_examples_history: i32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            num_iterations: 1000,
            num_episodes: 100,
            temp_threshold: 15,
            update_threshold: 0.6,
            max_moves: 50,
            num_mcts_sims: 25,
            arena_compare: 40,
            cpuct: 1.0,
            checkpoint_interval: 20,
            load_model: false,
            train_examples_history: 20,
        }
    }
}

#[pyfunction]
pub fn run_mcts_for_state(state_json: &str, simulations: usize) -> PyResult<String> {
    let state: GameState = serde_json::from_str(state_json)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let engine = Engine::from_state(state);
    let neural_net = NeuralNet::new(156, 96)?;
    let mut mcts = MCTS::new(1.0, neural_net);
    let probs = mcts.search(&engine, 0.0, simulations);
    let best_move = probs
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|(m, _)| m.get_move_index())
        .unwrap_or(0);
    Ok(serde_json::to_string(&best_move).unwrap())
}

#[pymodule]
fn klondike_ai(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_mcts_for_state, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_config_default() {
        let config = TrainingConfig::default();
        assert_eq!(config.num_iterations, 1000);
        assert_eq!(config.num_episodes, 100);
        assert_eq!(config.temp_threshold, 15);
        assert_eq!(config.update_threshold, 0.6);
        assert_eq!(config.max_moves, 50);
        assert_eq!(config.num_mcts_sims, 25);
        assert_eq!(config.arena_compare, 40);
        assert_eq!(config.cpuct, 1.0);
        assert_eq!(config.checkpoint_interval, 20);
        assert!(!config.load_model);
        assert_eq!(config.train_examples_history, 20);
    }
}