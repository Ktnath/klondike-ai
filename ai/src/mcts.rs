use std::collections::HashMap;
use std::f32;
use rand::seq::SliceRandom;
use crate::{Engine, Move, NeuralNet};

#[derive(Debug)]
struct Edge {
    prior: f32,
    visit_count: i32,
    total_action_value: f32,
    mean_action_value: f32,
}

impl Edge {
    fn new(prior: f32) -> Self {
        Self {
            prior,
            visit_count: 0,
            total_action_value: 0.0,
            mean_action_value: 0.0,
        }
    }

    fn update(&mut self, value: f32) {
        self.visit_count += 1;
        self.total_action_value += value;
        self.mean_action_value = self.total_action_value / self.visit_count as f32;
    }
}

pub struct MCTS {
    cpuct: f32,
    edges: HashMap<String, Vec<Edge>>,
    neural_net: NeuralNet,
}

impl MCTS {
    pub fn new(cpuct: f32, neural_net: NeuralNet) -> Self {
        Self {
            cpuct,
            edges: HashMap::new(),
            neural_net,
        }
    }

    pub fn search(&mut self, engine: &Engine, temp: f32, simulations: usize) -> Vec<(Move, f32)> {
        let state_str = self.get_state_str(engine);
        
        // Si l'état n'a pas été visité, l'évaluer avec le réseau de neurones
        if !self.edges.contains_key(&state_str) {
            let (pi, _) = self.evaluate(engine);
            let moves = engine.get_available_moves();
            let edges: Vec<Edge> = moves.iter()
                .map(|m| Edge::new(pi[m.get_move_index()]))
                .collect();
            self.edges.insert(state_str.clone(), edges);
        }

        // Effectuer les simulations MCTS
        for _ in 0..simulations {
            self.simulate(engine.clone());
        }

        // Calculer les probabilités de chaque coup
        let edges = self.edges.get(&state_str).unwrap();
        let moves = engine.get_available_moves();
        let mut probs: Vec<(Move, f32)> = moves.iter()
            .zip(edges.iter())
            .map(|(m, e)| (*m, (e.visit_count as f32).powf(1.0 / temp)))
            .collect();

        // Normaliser les probabilités
        let sum: f32 = probs.iter().map(|(_, p)| p).sum();
        for (_, p) in probs.iter_mut() {
            *p /= sum;
        }

        probs
    }

    fn simulate(&mut self, mut engine: Engine) -> f32 {
        let state_str = self.get_state_str(&engine);

        // Si l'état est terminal, retourner sa valeur
        if engine.get_state().is_won() {
            return 1.0;
        }

        // Si l'état n'a pas été visité, l'évaluer
        if !self.edges.contains_key(&state_str) {
            let (pi, v) = self.evaluate(&engine);
            let moves = engine.get_available_moves();
            let edges: Vec<Edge> = moves.iter()
                .map(|m| Edge::new(pi[m.get_move_index()]))
                .collect();
            self.edges.insert(state_str, edges);
            return v;
        }

        // Sélectionner le meilleur coup selon UCT
        let moves = engine.get_available_moves();
        let edges = self.edges.get(&state_str).unwrap();
        let total_visits: i32 = edges.iter().map(|e| e.visit_count).sum();

        let (best_move, best_edge) = moves.iter()
            .zip(edges.iter())
            .max_by(|(_, e1), (_, e2)| {
                let score1 = self.uct_score(e1, total_visits as f32);
                let score2 = self.uct_score(e2, total_visits as f32);
                score1.partial_cmp(&score2).unwrap()
            })
            .unwrap();

        // Effectuer le coup et récurser
        engine.make_move(best_move);
        let v = -self.simulate(engine);

        // Mettre à jour les statistiques
        let edges = self.edges.get_mut(&state_str).unwrap();
        let edge = &mut edges[best_move.get_move_index()];
        edge.update(v);

        v
    }

    fn evaluate(&self, engine: &Engine) -> (Vec<f32>, f32) {
        let board = engine.get_state().encode_observation();
        self.neural_net.predict(board).unwrap()
    }

    fn uct_score(&self, edge: &Edge, total_visits: f32) -> f32 {
        if edge.visit_count == 0 {
            return f32::INFINITY;
        }

        edge.mean_action_value +
            self.cpuct * edge.prior * (total_visits.sqrt() / (1.0 + edge.visit_count as f32))
    }

    fn get_state_str(&self, engine: &Engine) -> String {
        format!("{:?}", engine.get_state().encode_observation())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use klondike_core::Engine;

    #[test]
    fn test_mcts_creation() {
        let neural_net = NeuralNet::new(156, 96).unwrap();
        let mcts = MCTS::new(1.0, neural_net);
        assert_eq!(mcts.cpuct, 1.0);
        assert!(mcts.edges.is_empty());
    }

    #[test]
    fn test_mcts_search() {
        let neural_net = NeuralNet::new(156, 96).unwrap();
        let mut mcts = MCTS::new(1.0, neural_net);
        let engine = Engine::new();

        let probs = mcts.search(&engine, 1.0, 10);
        assert!(!probs.is_empty());
        
        let sum: f32 = probs.iter().map(|(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
}