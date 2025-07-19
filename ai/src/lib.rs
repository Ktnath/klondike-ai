mod network;
mod mcts;
mod coach;

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