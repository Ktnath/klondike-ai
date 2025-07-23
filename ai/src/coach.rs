use crate::{Engine, NeuralNet, TrainingConfig, MCTS};
use rand::{Rng, SeedableRng};
use std::collections::VecDeque;

pub struct Coach {
    neural_net: NeuralNet,
    config: TrainingConfig,
    training_examples: VecDeque<Vec<(Vec<f32>, Vec<f32>, f32)>>,
}

impl Coach {
    pub fn new(neural_net: NeuralNet, config: TrainingConfig) -> Self {
        Self {
            neural_net,
            config,
            training_examples: VecDeque::new(),
        }
    }

    pub fn learn(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        for i in 0..self.config.num_iterations {
            println!("Itération {}", i + 1);

            // Générer de nouveaux exemples d'entraînement
            let mut examples = Vec::new();
            for _ in 0..self.config.num_episodes {
                examples.extend(self.execute_episode());
            }

            // Ajouter les nouveaux exemples à l'historique
            self.training_examples.push_back(examples);
            if self.training_examples.len() > self.config.train_examples_history as usize {
                self.training_examples.pop_front();
            }

            // Fusionner tous les exemples pour l'entraînement
            let training_data: Vec<_> = self
                .training_examples
                .iter()
                .flat_map(|examples| examples.iter().cloned())
                .collect();

            // Entraîner le réseau
            self.neural_net.train(training_data)?;

            // Sauvegarder le modèle périodiquement
            if (i + 1) % self.config.checkpoint_interval == 0 {
                let folder = "models";
                let filename = format!("checkpoint_{}.pth", i + 1);
                std::fs::create_dir_all(folder)?;
                self.neural_net.save_checkpoint(folder, &filename)?;
            }
        }

        Ok(())
    }

    fn execute_episode(&self) -> Vec<(Vec<f32>, Vec<f32>, f32)> {
        let mut examples = Vec::new();
        let mut engine = Engine::new();
        let mut mcts = MCTS::new(self.config.cpuct, self.neural_net.clone());

        let mut move_count = 0;
        while !engine.get_state().is_won() && move_count < self.config.max_moves {
            let temp = if move_count < self.config.temp_threshold {
                1.0
            } else {
                0.0
            };

            // Obtenir les probabilités des coups via MCTS
            let state = engine.get_state().encode_observation();
            let probs = mcts.search(&engine, temp, self.config.num_mcts_sims as usize);

            // Sauvegarder l'état actuel
            let mut pi = vec![0.0; 96]; // Taille de l'espace d'actions
            for (mov, prob) in probs.iter() {
                pi[mov.get_move_index()] = *prob;
            }
            examples.push((state, pi.clone(), 0.0)); // La valeur sera mise à jour plus tard

            // Sélectionner et jouer un coup
            let selected_move = if temp == 0.0 {
                // Choisir le meilleur coup
                probs
                    .iter()
                    .max_by(|(_, p1), (_, p2)| p1.partial_cmp(p2).unwrap())
                    .map(|(m, _)| m.clone())
                    .unwrap()
            } else {
                // Échantillonner selon les probabilités
                let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
                #[allow(deprecated)]
                let r: f32 = rng.gen();
                let mut sum = 0.0;
                let mut selected = probs[0].0.clone();
                for (mov, prob) in probs.iter() {
                    sum += prob;
                    if sum > r {
                        selected = mov.clone();
                        break;
                    }
                }
                selected
            };

            engine.make_move(&selected_move);
            move_count += 1;
        }

        // Mettre à jour les valeurs des exemples
        let value = if engine.get_state().is_won() {
            1.0
        } else {
            0.0 // Match nul ou limite de coups atteinte
        };

        for example in examples.iter_mut() {
            example.2 = value;
        }

        examples
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coach_creation() {
        let neural_net = NeuralNet::new(156, 96).unwrap();
        let config = TrainingConfig::default();
        let coach = Coach::new(neural_net, config);
        assert!(coach.training_examples.is_empty());
    }

    #[test]
    fn test_execute_episode() {
        let neural_net = NeuralNet::new(156, 96).unwrap();
        let config = TrainingConfig::default();
        let coach = Coach::new(neural_net, config);

        let examples = coach.execute_episode();
        assert!(!examples.is_empty());

        for (state, pi, value) in examples {
            assert_eq!(state.len(), 156);
            assert_eq!(pi.len(), 96);
            assert!(value >= 0.0 && value <= 1.0);
        }
    }
}
