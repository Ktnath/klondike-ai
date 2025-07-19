mod card;
mod state;
mod moves;
mod engine;

pub use card::Card;
pub use state::GameState;
pub use moves::Move;
pub use engine::Engine;

/// Nombre de cartes dans un jeu
pub const N_CARDS: u32 = 52;
/// Nombre de piles de tableau
pub const N_PILES: u32 = 7;
/// Nombre de fondations
pub const N_FOUNDATIONS: u32 = 4;
/// Nombre de couleurs
pub const N_SUITS: u32 = 4;

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