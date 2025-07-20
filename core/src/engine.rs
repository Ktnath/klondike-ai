use pyo3::prelude::*;
use crate::{GameState, Move};

#[pyclass]
pub struct Engine {
    state: GameState,
    available_moves: Vec<Move>,
}

impl Engine {
    pub fn new() -> Self {
        let state = GameState::new();
        let mut engine = Self {
            state,
            available_moves: Vec::new(),
        };
        engine.update_available_moves();
        engine
    }

    pub fn from_state(state: GameState) -> Self {
        let mut engine = Self {
            state,
            available_moves: Vec::new(),
        };
        engine.update_available_moves();
        engine
    }

    pub fn get_state(&self) -> &GameState {
        &self.state
    }

    pub fn get_available_moves(&self) -> &[Move] {
        &self.available_moves
    }

    pub fn make_move(&mut self, game_move: &Move) -> bool {
        if !self.is_move_legal(game_move) {
            return false;
        }

        match *game_move {
            Move::DrawCard => {
                if !self.state.draw_card() {
                    return false;
                }
            },
            Move::WasteToTableau { destination } => {
                if let Some(card) = self.state.get_top_waste() {
                    if let Some(stack) = self.state.get_tableau_stack(destination) {
                        if !stack.is_empty() && !card.can_stack_on(stack.last().unwrap()) {
                            return false;
                        }
                    }
                } else {
                    return false;
                }
                // Implémentation du mouvement...
            },
            Move::WasteToFoundation { foundation } => {
                if let Some(card) = self.state.get_top_waste() {
                    if !card.can_place_on_foundation(
                        self.state.get_foundation_top(foundation)
                    ) {
                        return false;
                    }
                } else {
                    return false;
                }
                // Implémentation du mouvement...
            },
            Move::TableauToTableau { from, to, count } => {
                if let Some(source) = self.state.get_tableau_stack(from) {
                    if source.len() < count {
                        return false;
                    }
                    let start_idx = source.len() - count;
                    if !source[start_idx].is_face_up() {
                        return false;
                    }
                    if let Some(dest) = self.state.get_tableau_stack(to) {
                        if !dest.is_empty() && 
                           !source[start_idx].can_stack_on(dest.last().unwrap()) {
                            return false;
                        }
                    }
                } else {
                    return false;
                }
                // Implémentation du mouvement...
            },
            Move::TableauToFoundation { from, foundation } => {
                if let Some(source) = self.state.get_tableau_stack(from) {
                    if source.is_empty() {
                        return false;
                    }
                    if !source.last().unwrap().can_place_on_foundation(
                        self.state.get_foundation_top(foundation)
                    ) {
                        return false;
                    }
                } else {
                    return false;
                }
                // Implémentation du mouvement...
            },
            Move::FlipTableauCard { pile } => {
                if let Some(stack) = self.state.get_tableau_stack(pile) {
                    if stack.is_empty() || stack.last().unwrap().is_face_up() {
                        return false;
                    }
                } else {
                    return false;
                }
                // Implémentation du mouvement...
            },
        }

        self.update_available_moves();
        true
    }

    fn is_move_legal(&self, game_move: &Move) -> bool {
        self.available_moves.contains(game_move)
    }

    fn update_available_moves(&mut self) {
        self.available_moves.clear();

        // Vérifier si on peut piocher
        if !self.state.get_top_waste().is_some() {
            self.available_moves.push(Move::DrawCard);
        }

        // Vérifier les mouvements possibles depuis la défausse
        if let Some(waste_card) = self.state.get_top_waste() {
            // Vers les piles du tableau
            for dest in 0..7 {
                if let Some(stack) = self.state.get_tableau_stack(dest) {
                    if stack.is_empty() || waste_card.can_stack_on(stack.last().unwrap()) {
                        self.available_moves.push(Move::WasteToTableau { destination: dest });
                    }
                }
            }

            // Vers les fondations
            for foundation in 0..4 {
                if waste_card.can_place_on_foundation(
                    self.state.get_foundation_top(foundation)
                ) {
                    self.available_moves.push(Move::WasteToFoundation { foundation });
                }
            }
        }

        // Vérifier les mouvements entre piles du tableau
        for from in 0..7 {
            if let Some(source) = self.state.get_tableau_stack(from) {
                if source.is_empty() {
                    continue;
                }

                // Vers d'autres piles
                for to in 0..7 {
                    if from == to {
                        continue;
                    }
                    if let Some(dest) = self.state.get_tableau_stack(to) {
                        if dest.is_empty() || source.last().unwrap().can_stack_on(dest.last().unwrap()) {
                            for count in 1..=source.len() {
                                if source[source.len() - count].is_face_up() {
                                    self.available_moves.push(Move::TableauToTableau {
                                        from,
                                        to,
                                        count,
                                    });
                                } else {
                                    break;
                                }
                            }
                        }
                    }
                }

                // Vers les fondations
                for foundation in 0..4 {
                    if source.last().unwrap().can_place_on_foundation(
                        self.state.get_foundation_top(foundation)
                    ) {
                        self.available_moves.push(Move::TableauToFoundation {
                            from,
                            foundation,
                        });
                    }
                }
            }

            // Vérifier si on peut retourner une carte
            if let Some(stack) = self.state.get_tableau_stack(from) {
                if !stack.is_empty() && !stack.last().unwrap().is_face_up() {
                    self.available_moves.push(Move::FlipTableauCard { pile: from });
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_initialization() {
        let engine = Engine::new();
        assert!(!engine.get_available_moves().is_empty());
    }

    #[test]
    fn test_draw_card() {
        let mut engine = Engine::new();
        if engine.get_available_moves().contains(&Move::DrawCard) {
            assert!(engine.make_move(&Move::DrawCard));
        }
    }

    #[test]
    fn test_illegal_move() {
        let mut engine = Engine::new();
        assert!(!engine.make_move(&Move::WasteToTableau { destination: 0 }));
    }
}