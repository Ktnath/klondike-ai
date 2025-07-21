use pyo3::prelude::*;
use serde::{Serialize, Deserialize};
use crate::{Card, N_CARDS, N_PILES, N_FOUNDATIONS};

#[pyclass]
#[derive(Clone, Serialize, Deserialize)]
pub struct GameState {
    // Cartes dans le talon (pioche)
    stock: Vec<Card>,
    // Cartes retournées du talon
    waste: Vec<Card>,
    // Piles de tableau
    tableau: [Vec<Card>; N_PILES as usize],
    // Piles de fondation
    foundations: [Vec<Card>; N_FOUNDATIONS as usize],
    // Score actuel
    score: i32,
}

impl GameState {
    pub fn new() -> Self {
        let mut cards: Vec<Card> = (0..N_CARDS)
            .map(|i| Card::new(i as u8 % 13, i as u8 / 13))
            .collect();

        use rand::seq::SliceRandom;
        use rand::SeedableRng;
        use rand::rngs::SmallRng;
        let mut rng = SmallRng::seed_from_u64(0);
        cards.shuffle(&mut rng);

        let mut tableau = [Vec::new(), Vec::new(), Vec::new(), Vec::new(),
                          Vec::new(), Vec::new(), Vec::new()];
        
        // Distribution initiale des cartes
        let mut card_index = 0;
        for pile in 0..N_PILES {
            for position in 0..=pile {
                let mut card = cards[card_index];
                if position == pile {
                    card.flip();
                }
                tableau[pile as usize].push(card);
                card_index += 1;
            }
        }

        // Le reste des cartes va dans le talon
        let stock = cards[card_index..].to_vec();

        Self {
            stock,
            waste: Vec::new(),
            tableau,
            foundations: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            score: 0,
        }
    }

    pub fn draw_card(&mut self) -> bool {
        if self.stock.is_empty() {
            if self.waste.is_empty() {
                return false;
            }
            // Retourner le talon
            self.stock = self.waste.drain(..).rev().collect();
            return true;
        }

        let mut card = self.stock.pop().unwrap();
        card.flip();
        self.waste.push(card);
        true
    }

    pub fn get_top_waste(&self) -> Option<&Card> {
        self.waste.last()
    }

    pub fn get_tableau_stack(&self, pile: usize) -> Option<&[Card]> {
        if pile >= N_PILES as usize {
            return None;
        }
        Some(&self.tableau[pile])
    }

    pub fn get_foundation_top(&self, foundation: usize) -> Option<&Card> {
        if foundation >= N_FOUNDATIONS as usize {
            return None;
        }
        self.foundations[foundation].last()
    }

    pub fn is_won(&self) -> bool {
        self.foundations.iter().all(|f| f.len() == 13)
    }

    pub fn get_score(&self) -> i32 {
        self.score
    }

    pub fn foundation_count(&self) -> usize {
        self.foundations.iter().map(|f| f.len()).sum()
    }

    pub fn encode_observation(&self) -> Vec<f32> {
        let mut obs = Vec::with_capacity(N_CARDS as usize * 2);
        
        // Encoder chaque carte avec sa position et son état (face visible/cachée)
        for pile in 0..N_PILES as usize {
            for card in &self.tableau[pile] {
                obs.push(card.rank() as f32 / 12.0);
                obs.push(card.suit() as f32 / 3.0);
                obs.push(if card.is_face_up() { 1.0 } else { 0.0 });
            }
        }

        // Encoder les fondations
        for foundation in &self.foundations {
            if let Some(top) = foundation.last() {
                obs.push(top.rank() as f32 / 12.0);
                obs.push(top.suit() as f32 / 3.0);
                obs.push(1.0); // Toujours face visible
            }
        }

        // Encoder le talon et la défausse
        if let Some(top) = self.get_top_waste() {
            obs.push(top.rank() as f32 / 12.0);
            obs.push(top.suit() as f32 / 3.0);
            obs.push(1.0);
        }

        obs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_game_initialization() {
        let game = GameState::new();
        
        // Vérifier la distribution initiale
        for (i, pile) in game.tableau.iter().enumerate() {
            assert_eq!(pile.len(), i + 1);
            assert!(pile.last().unwrap().is_face_up());
            assert!(pile[..i].iter().all(|card| !card.is_face_up()));
        }

        // Vérifier que toutes les cartes sont présentes
        let total_cards = game.stock.len() 
            + game.waste.len() 
            + game.tableau.iter().map(|p| p.len()).sum::<usize>()
            + game.foundations.iter().map(|f| f.len()).sum::<usize>();
        assert_eq!(total_cards, N_CARDS as usize);
    }

    #[test]
    fn test_draw_card() {
        let mut game = GameState::new();
        let initial_stock_size = game.stock.len();
        
        assert!(game.draw_card());
        assert_eq!(game.stock.len(), initial_stock_size - 1);
        assert_eq!(game.waste.len(), 1);
        assert!(game.waste.last().unwrap().is_face_up());
    }

    #[test]
    fn test_encode_observation() {
        let game = GameState::new();
        let obs = game.encode_observation();
        assert!(!obs.is_empty());
        assert!(obs.iter().all(|&x| x >= 0.0 && x <= 1.0));
    }
}