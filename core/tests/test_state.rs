use core::num::NonZeroU8;

use klondike_core::{deck::N_PILES, shuffler::default_shuffle, state::Solitaire};

#[test]
fn compute_visible_piles_matches_hidden_top() {
    let game = Solitaire::new(&default_shuffle(0), NonZeroU8::new(1).unwrap());
    let piles = game.compute_visible_piles();

    for pos in 0..N_PILES {
        let top = game.get_hidden().peek(pos).copied().unwrap();
        assert_eq!(piles[pos as usize][0], top);
    }
}

#[test]
fn compute_visible_piles_descends_alternating() {
    let game = Solitaire::new(&default_shuffle(0), NonZeroU8::new(1).unwrap());
    let piles = game.compute_visible_piles();
    for pile in &piles {
        for w in pile.windows(2) {
            assert!(w[1].go_after(Some(w[0])));
        }
    }
}

#[test]
fn is_valid_true_for_new_game() {
    let game = Solitaire::new(&default_shuffle(0), NonZeroU8::new(1).unwrap());
    assert!(game.is_valid());
}
