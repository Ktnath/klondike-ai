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

#[pyfunction]
pub fn new_game(seed: Option<&str>) -> PyResult<String> {
    let seed_val = seed.and_then(|s| s.parse::<u64>().ok()).unwrap_or(0);
    let cards = default_shuffle(seed_val);
    let game = Solitaire::new(&cards, NonZeroU8::new(1).unwrap());
    Ok(encode_state(&game))
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
    let mut st = decode_state(state)
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("invalid state"))?;
    let Some(m) = parse_move(mv) else {
        return Ok((encode_state(&st), false));
    };
    let mut engine: SolitaireEngine<crate::pruning::FullPruner> = st.into();
    let valid = engine.do_move(m);
    st = engine.into_state();
    Ok((encode_state(&st), valid))
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
    let parsed: Value = serde_json::from_str(state)
        .map_err(|e| PyValueError::new_err(format!("Invalid JSON: {}", e)))?;

    let foundations = match parsed.get("foundations").and_then(|f| f.as_array()) {
        Some(f) => f,
        None => {
            eprintln!("⚠️ compute_base_reward_json: missing 'foundations' field");
            return Ok(0.0);
        }
    };

    let total_cards = foundations
        .iter()
        .filter_map(|pile| pile.as_array())
        .map(|pile| pile.len())
        .sum::<usize>();

    Ok(total_cards as f32)
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
    m.add_function(wrap_pyfunction!(compute_base_reward_json, m)?)?;
    Ok(())
}
