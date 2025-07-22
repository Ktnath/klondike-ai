// Use the standard library when building the Python extension. The previous
// implementation attempted to compile with `no_std` which requires a global
// allocator and a panic handler. For simplicity and to allow integration with
// PyO3 we rely on the standard library.

pub mod analysis;
pub mod card;
pub mod convert;
pub mod deck;
pub mod dependencies;
pub mod engine;
pub mod formatter;
pub mod game_theory;
pub mod graph;
pub mod hidden;
pub mod hop_solver;
pub mod mcts_solver;
pub mod moves;
pub mod partial;
pub mod pruning;
pub mod shuffler;
pub mod solver;
pub mod stack;
pub mod standard;
pub mod state;
pub mod tracking;
pub mod traverse;
mod utils;

use crate::engine::SolitaireEngine;
use crate::moves::Move;
use crate::shuffler::default_shuffle;
use crate::state::Solitaire;
use core::num::NonZeroU8;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use serde_json::Value;

fn encode_state(game: &Solitaire) -> String {
    format!("{}", game.encode())
}

fn decode_state(data: &str) -> Option<Solitaire> {
    let enc: u64 = data.parse().ok()?;
    let mut rng = SmallRng::seed_from_u64(0);
    let mut st = Solitaire::deal_with_rng(&mut rng);
    st.decode(enc);
    Some(st)
}

fn parse_move(data: &str) -> Option<Move> {
    // very small parser expecting format "<TYPE> <idx>"
    let mut parts = data.split_whitespace();
    let t = parts.next()?;
    let idx: u8 = parts.next()?.parse().ok()?;
    Some(match t {
        "DS" => Move::DeckStack(crate::card::Card::from_mask_index(idx)),
        "PS" => Move::PileStack(crate::card::Card::from_mask_index(idx)),
        "DP" => Move::DeckPile(crate::card::Card::from_mask_index(idx)),
        "SP" => Move::StackPile(crate::card::Card::from_mask_index(idx)),
        "R" => Move::Reveal(crate::card::Card::from_mask_index(idx)),
        _ => return None,
    })
}

fn move_to_string(m: Move) -> String {
    match m {
        Move::DeckStack(c) => format!("DS {}", c.mask_index()),
        Move::PileStack(c) => format!("PS {}", c.mask_index()),
        Move::DeckPile(c) => format!("DP {}", c.mask_index()),
        Move::StackPile(c) => format!("SP {}", c.mask_index()),
        Move::Reveal(c) => format!("R {}", c.mask_index()),
    }
}

fn card_to_string(card: crate::card::Card, lower: bool) -> String {
    use crate::formatter::NUMBERS;
    let (rank, suit) = card.split();
    let s = match suit {
        0 => 'H',
        1 => 'D',
        2 => 'C',
        3 => 'S',
        _ => 'x',
    };
    let suit_char = if lower { s.to_ascii_lowercase() } else { s };
    format!("{}{}", NUMBERS[rank as usize], suit_char)
}

#[pyfunction]
pub fn encode_state_to_json(encoded: &str) -> PyResult<String> {
    let st = decode_state(encoded)
        .ok_or_else(|| PyValueError::new_err("invalid state"))?;
    let game: crate::standard::StandardSolitaire = (&st).into();

    let tableau: Vec<Vec<String>> = (0..crate::deck::N_PILES)
        .map(|i| {
            let mut col = Vec::new();
            for c in &game.get_hidden()[i as usize] {
                col.push(card_to_string(*c, true));
            }
            for c in &game.get_piles()[i as usize] {
                col.push(card_to_string(*c, false));
            }
            col
        })
        .collect();

    let stock: Vec<String> = game
        .get_deck()
        .deck_iter()
        .rev()
        .map(|c| card_to_string(c, false))
        .collect();

    let waste: Vec<String> = game
        .get_deck()
        .waste_iter()
        .map(|c| card_to_string(c, false))
        .collect();

    let foundations: Vec<Vec<String>> = (0..crate::card::N_SUITS)
        .map(|s| {
            (0..game.get_stack().get(s))
                .map(|r| card_to_string(crate::card::Card::new(r, s), false))
                .collect()
        })
        .collect();

    let mut engine: SolitaireEngine<crate::pruning::FullPruner> = st.into();
    let moves = engine
        .list_moves()
        .iter()
        .map(|&m| move_to_string(m))
        .collect::<Vec<_>>();

    let json = serde_json::json!({
        "tableau": tableau,
        "foundations": foundations,
        "stock": stock,
        "waste": waste,
        "score": 0,
        "moves": moves,
        "encoded": encoded
    });

    serde_json::to_string(&json).map_err(|e| PyValueError::new_err(e.to_string()))
}


#[pyfunction]
pub fn new_game(seed: Option<&str>) -> PyResult<String> {
    let seed_val = seed.and_then(|s| s.parse::<u64>().ok()).unwrap_or(0);
    let cards = default_shuffle(seed_val);
    let game = Solitaire::new(&cards, NonZeroU8::new(1).unwrap());
    let encoded = encode_state(&game);
    encode_state_to_json(&encoded)
}

#[pyfunction]
pub fn legal_moves(state: &str) -> PyResult<Vec<String>> {
    let st = decode_state(state)
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("invalid state"))?;
    let engine: SolitaireEngine<crate::pruning::FullPruner> = st.into();
    let moves = engine.list_moves();
    Ok(moves.iter().map(|&m| move_to_string(m)).collect())
}

#[pyfunction]
pub fn play_move(state: &str, mv: &str) -> PyResult<(String, bool)> {
    let v: Value = serde_json::from_str(state)
        .map_err(|_| PyValueError::new_err("Invalid state"))?;
    let encoded = v.get("encoded")
        .and_then(|e| e.as_str())
        .ok_or_else(|| PyValueError::new_err("Missing encoded field"))?;
    let mut st = decode_state(encoded)
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("invalid state"))?;
    let Some(m) = parse_move(mv) else {
        let encoded = encode_state(&st);
        let json = encode_state_to_json(&encoded)?;
        return Ok((json, false));
    };
    let mut engine: SolitaireEngine<crate::pruning::FullPruner> = st.into();
    let valid = engine.do_move(m);
    st = engine.into_state();
    let encoded = encode_state(&st);
    let json = encode_state_to_json(&encoded)?;
    Ok((json, valid))
}

#[pyfunction]
pub fn encode_observation(_state: &str) -> PyResult<Vec<f32>> {
    // Placeholder observation vector used by Python code.
    Ok(vec![0.0; 156])
}

#[pyfunction]
pub fn foundation_count(state: &str) -> PyResult<usize> {
    let st = decode_state(state)
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("invalid state"))?;
    let mut count = 0usize;
    for suit in 0..crate::card::N_SUITS {
        count += st.get_stack().get(suit) as usize;
    }
    Ok(count)
}

#[pyfunction]
pub fn is_won(state: &str) -> PyResult<bool> {
    let st = decode_state(state)
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("invalid state"))?;
    Ok(st.is_win())
}

#[pyfunction]
pub fn compute_base_reward_json(state: &str) -> PyResult<f32> {
    let v: Value = serde_json::from_str(state)
        .map_err(|_| PyValueError::new_err("Invalid JSON passed to reward engine"))?;

    if !v.get("foundations").is_some() {
        return Err(PyValueError::new_err("Missing 'foundations' field"));
    }

    let count = v["foundations"]
        .as_array()
        .ok_or_else(|| PyValueError::new_err("'foundations' should be an array"))?
        .iter()
        .map(|stack| stack.as_array().map(|a| a.len()).unwrap_or(0))
        .sum::<usize>();

    Ok(count as f32 / 52.0)
}

#[pyfunction]
pub fn move_index(_mv: &str) -> PyResult<usize> {
    // Simplistic constant mapping used for testing
    Ok(0)
}

#[pyfunction]
pub fn move_from_index(_idx: usize) -> PyResult<Option<String>> {
    Ok(None)
}

#[pyfunction]
pub fn solve_klondike(_seed: &str) -> PyResult<String> {
    Ok("{\"result\":null}".to_string())
}

#[pyfunction]
pub fn shuffle_seed() -> PyResult<u64> {
    Ok(0)
}

#[pymodule]
fn klondike_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(new_game, m)?)?;
    m.add_function(wrap_pyfunction!(legal_moves, m)?)?;
    m.add_function(wrap_pyfunction!(play_move, m)?)?;
    m.add_function(wrap_pyfunction!(encode_observation, m)?)?;
    m.add_function(wrap_pyfunction!(foundation_count, m)?)?;
    m.add_function(wrap_pyfunction!(is_won, m)?)?;
    m.add_function(wrap_pyfunction!(move_index, m)?)?;
    m.add_function(wrap_pyfunction!(move_from_index, m)?)?;
    m.add_function(wrap_pyfunction!(solve_klondike, m)?)?;
    m.add_function(wrap_pyfunction!(shuffle_seed, m)?)?;
    m.add_function(wrap_pyfunction!(encode_state_to_json, m)?)?;
    m.add_function(wrap_pyfunction!(compute_base_reward_json, m)?)?;
    Ok(())
}
