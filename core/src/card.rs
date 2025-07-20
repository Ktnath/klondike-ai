use pyo3::prelude::*;

use serde::{Serialize, Deserialize};

#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Card {
    rank: u8,
    suit: u8,
    is_face_up: bool,
}

impl Card {
    pub fn new(rank: u8, suit: u8) -> Self {
        assert!(rank < 13, "Le rang doit être entre 0 et 12");
        assert!(suit < 4, "La couleur doit être entre 0 et 3");
        Self {
            rank,
            suit,
            is_face_up: false,
        }
    }

    pub fn rank(&self) -> u8 {
        self.rank
    }

    pub fn suit(&self) -> u8 {
        self.suit
    }

    pub fn is_face_up(&self) -> bool {
        self.is_face_up
    }

    pub fn flip(&mut self) {
        self.is_face_up = !self.is_face_up;
    }

    pub fn is_red(&self) -> bool {
        self.suit == 1 || self.suit == 2 // Cœurs ou Carreaux
    }

    pub fn is_black(&self) -> bool {
        self.suit == 0 || self.suit == 3 // Piques ou Trèfles
    }

    pub fn can_stack_on(&self, other: &Card) -> bool {
        if !self.is_face_up || !other.is_face_up {
            return false;
        }
        self.rank == other.rank - 1 && self.is_red() != other.is_red()
    }

    pub fn can_place_on_foundation(&self, foundation_top: Option<&Card>) -> bool {
        if !self.is_face_up {
            return false;
        }
        match foundation_top {
            None => self.rank == 0, // As
            Some(top) => {
                self.suit == top.suit && self.rank == top.rank + 1
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_card_creation() {
        let card = Card::new(0, 0);
        assert_eq!(card.rank(), 0);
        assert_eq!(card.suit(), 0);
        assert!(!card.is_face_up());
    }

    #[test]
    fn test_card_color() {
        let spade = Card::new(0, 0);
        let heart = Card::new(0, 1);
        assert!(spade.is_black());
        assert!(heart.is_red());
    }

    #[test]
    fn test_card_stacking() {
        let mut black_king = Card::new(12, 0);
        let mut red_queen = Card::new(11, 1);
        black_king.flip();
        red_queen.flip();
        assert!(red_queen.can_stack_on(&black_king));
    }

    #[test]
    fn test_foundation_placement() {
        let mut ace_spades = Card::new(0, 0);
        let mut two_spades = Card::new(1, 0);
        ace_spades.flip();
        two_spades.flip();
        assert!(ace_spades.can_place_on_foundation(None));
        assert!(two_spades.can_place_on_foundation(Some(&ace_spades)));
    }
}