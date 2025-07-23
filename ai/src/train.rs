use crate::{Coach, NeuralNet, TrainingConfig};

/// Entry point for a basic self-play training loop.
/// The original complex training module relied on removed engine
/// structures, so this simplified version only wires the coach and
/// neural network together.
pub fn train_default() -> Result<(), Box<dyn std::error::Error>> {
    let net = NeuralNet::new(156, 96)?;
    let mut coach = Coach::new(net, TrainingConfig::default());
    coach.learn()
}
