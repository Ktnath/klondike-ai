use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use serde_json::Value;

use lonelybot::card::Card;
use lonelybot::engine::SolitaireEngine;
use lonelybot::moves::Move as EngineMove;
use lonelybot::pruning::FullPruner;
use lonelybot::shuffler::default_shuffle;
use lonelybot::solver::{solve_with_tracking, SearchResult};
use lonelybot::standard::StandardSolitaire;
use lonelybot::state::Solitaire;
use lonelybot::tracking::{DefaultTerminateSignal, EmptySearchStats};
use core::num::NonZeroU8;

fn encode_state(game: &Solitaire) -> String {
    format!("{}", game.encode())
}

fn decode_state(data: &str) -> Option<Solitaire> {
    let enc: u64 = data.parse().ok()?;
    Some(Solitaire::from_encode(enc))
}

fn parse_move(data: &str) -> Option<EngineMove> {
    let mut parts = data.split_whitespace();
    let t = parts.next()?;
    let idx: u8 = parts.next()?.parse().ok()?;
    Some(match t {
        "DS" => EngineMove::DeckStack(Card::from_mask_index(idx)),
        "PS" => EngineMove::PileStack(Card::from_mask_index(idx)),
        "DP" => EngineMove::DeckPile(Card::from_mask_index(idx)),
        "SP" => EngineMove::StackPile(Card::from_mask_index(idx)),
        "R" => EngineMove::Reveal(Card::from_mask_index(idx)),
        _ => return None,
    })
}

fn move_to_string(m: EngineMove) -> String {
    match m {
        EngineMove::DeckStack(c) => format!("DS {}", c.mask_index()),
        EngineMove::PileStack(c) => format!("PS {}", c.mask_index()),
        EngineMove::DeckPile(c) => format!("DP {}", c.mask_index()),
        EngineMove::StackPile(c) => format!("SP {}", c.mask_index()),
        EngineMove::Reveal(c) => format!("R {}", c.mask_index()),
    }
}

fn card_to_string(card: Card, lower: bool) -> String {
    use lonelybot::formatter::NUMBERS;
    let (rank, suit) = card.split();
    let s = match suit { 0 => 'H', 1 => 'D', 2 => 'C', 3 => 'S', _ => 'x' };
    let suit_char = if lower { s.to_ascii_lowercase() } else { s };
    format!("{}{}", NUMBERS[rank as usize], suit_char)
}

#[pyfunction]
pub fn encode_state_to_json(encoded: &str) -> PyResult<String> {
    let st = decode_state(encoded).ok_or_else(|| PyValueError::new_err("invalid state"))?;
    let game: StandardSolitaire = (&st).into();

    let tableau: Vec<Vec<String>> = (0..lonelybot::deck::N_PILES)
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

    let foundations: Vec<Vec<String>> = (0..lonelybot::card::N_SUITS)
        .map(|s| {
            (0..game.get_stack().get(s))
                .map(|r| card_to_string(Card::new(r, s), false))
                .collect()
        })
        .collect();

    let engine: SolitaireEngine<FullPruner> = st.into();
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
        .ok_or_else(|| PyValueError::new_err("invalid state"))?;
    let engine: SolitaireEngine<FullPruner> = st.into();
    Ok(engine.list_moves().iter().map(|&m| move_to_string(m)).collect())
}

#[pyfunction]
pub fn play_move(state: &str, mv: &str) -> PyResult<(String, bool)> {
    let v: Value = serde_json::from_str(state).map_err(|_| PyValueError::new_err("Invalid state"))?;
    let encoded = v.get("encoded").and_then(|e| e.as_str()).ok_or_else(|| PyValueError::new_err("Missing encoded field"))?;
    let mut st = decode_state(encoded).ok_or_else(|| PyValueError::new_err("invalid state"))?;
    let Some(m) = parse_move(mv) else {
        let encoded = encode_state(&st);
        let json = encode_state_to_json(&encoded)?;
        return Ok((json, false));
    };
    let mut engine: SolitaireEngine<FullPruner> = st.into();
    let valid = engine.do_move(m);
    st = engine.into_state();
    let encoded = encode_state(&st);
    let json = encode_state_to_json(&encoded)?;
    Ok((json, valid))
}

#[pyfunction]
pub fn move_index(mv: &str) -> PyResult<usize> {
    let Some(m) = parse_move(mv) else {
        return Err(PyValueError::new_err("invalid move"));
    };
    let idx = match m {
        EngineMove::DeckStack(c) => c.mask_index() as usize,
        EngineMove::PileStack(c) => 52 + c.mask_index() as usize,
        EngineMove::DeckPile(c) => 104 + c.mask_index() as usize,
        EngineMove::StackPile(c) => 156 + c.mask_index() as usize,
        EngineMove::Reveal(c) => 208 + c.mask_index() as usize,
    };
    Ok(idx)
}

#[pyfunction]
pub fn move_from_index(index: usize) -> PyResult<Option<String>> {
    if index >= 260 {
        return Ok(None);
    }
    let ty = index / 52;
    let card_idx = (index % 52) as u8;
    let card = Card::from_mask_index(card_idx);
    let mv = match ty {
        0 => EngineMove::DeckStack(card),
        1 => EngineMove::PileStack(card),
        2 => EngineMove::DeckPile(card),
        3 => EngineMove::StackPile(card),
        _ => EngineMove::Reveal(card),
    };
    Ok(Some(move_to_string(mv)))
}

#[pyfunction]
pub fn shuffle_seed() -> PyResult<u64> {
    use std::time::{SystemTime, UNIX_EPOCH};
    if let Ok(seed) = std::env::var("SHUFFLE_SEED") {
        let parsed = seed.parse::<u64>().map_err(|_| PyValueError::new_err("invalid SHUFFLE_SEED"))?;
        Ok(parsed)
    } else {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().subsec_nanos();
        Ok((nanos as u64) ^ 0x5eED_F00Du64)
    }
}

#[pyfunction]
pub fn solve_klondike(state_json: &str) -> PyResult<Vec<(String, String)>> {
    let v: Value = serde_json::from_str(state_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let encoded = v.get("encoded").and_then(|e| e.as_str()).ok_or_else(|| PyValueError::new_err("missing encoded"))?;
    let mut state = decode_state(encoded).ok_or_else(|| PyValueError::new_err("invalid state"))?;
    let (status, history) = solve_with_tracking(&mut state, &EmptySearchStats {}, &DefaultTerminateSignal {});
    if status != SearchResult::Solved {
        return Ok(Vec::new());
    }
    let hist = history.ok_or_else(|| PyValueError::new_err("no history"))?;
    Ok(hist.iter().map(|m| (move_to_string(*m), String::new())).collect())
}

#[pyfunction]
pub fn is_won(state: &str) -> PyResult<bool> {
    let st = decode_state(state).ok_or_else(|| PyValueError::new_err("invalid state"))?;
    Ok(st.is_win())
}

#[pyfunction]
pub fn is_lost(state: &str) -> PyResult<bool> {
    let st = decode_state(state).ok_or_else(|| PyValueError::new_err("invalid state"))?;
    let engine: SolitaireEngine<FullPruner> = st.into();
    let legal = engine.list_moves();
    Ok(legal.is_empty() && !engine.state().is_win())
}

#[pyfunction]
pub fn compute_base_reward_json(state: &str) -> PyResult<f32> {
    let v: Value = serde_json::from_str(state).map_err(|_| PyValueError::new_err("Invalid JSON passed to reward engine"))?;
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
pub fn encode_observation(state: &str) -> PyResult<Vec<f32>> {
    // The implementation is simplified: reuse base reward computation to build observation.
    // For now return empty vector.
    let _ = state; // suppress unused
    Ok(Vec::new())
}

#[pymodule]
fn klondike_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(encode_state_to_json, m)?)?;
    m.add_function(wrap_pyfunction!(new_game, m)?)?;
    m.add_function(wrap_pyfunction!(legal_moves, m)?)?;
    m.add_function(wrap_pyfunction!(play_move, m)?)?;
    m.add_function(wrap_pyfunction!(move_index, m)?)?;
    m.add_function(wrap_pyfunction!(move_from_index, m)?)?;
    m.add_function(wrap_pyfunction!(shuffle_seed, m)?)?;
    m.add_function(wrap_pyfunction!(solve_klondike, m)?)?;
    m.add_function(wrap_pyfunction!(is_won, m)?)?;
    m.add_function(wrap_pyfunction!(is_lost, m)?)?;
    m.add_function(wrap_pyfunction!(compute_base_reward_json, m)?)?;
    m.add_function(wrap_pyfunction!(encode_observation, m)?)?;
    Ok(())
}

