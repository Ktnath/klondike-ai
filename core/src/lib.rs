// Use the standard library when building the Python extension. The previous
// implementation attempted to compile with `no_std` which requires a global
// allocator and a panic handler. For simplicity and to allow integration with
// PyO3 we rely on the standard library.

pub mod analysis;
pub mod card;
pub mod convert;
pub mod deck;
pub mod engine;
pub mod formatter;
pub mod game_theory;
pub mod hidden;
pub mod legacy; // LEGACY modules
                // Re-export legacy solver and graph modules for backward compatibility
pub use crate::legacy::graph;
pub use crate::legacy::solver;
pub mod intentions;
pub mod mcts_solver;
pub mod moves;
pub mod partial;
pub mod pruning;
pub mod shuffler;
pub mod stack;
pub mod standard;
pub mod state;
pub mod tracking;
pub mod traverse;
mod utils;

use crate::engine::SolitaireEngine;
use crate::legacy::solver::{solve_with_tracking, SearchResult};
use crate::moves::Move as EngineMove;
use crate::pruning::FullPruner;
use crate::shuffler::default_shuffle;
use crate::state::Solitaire;
use crate::tracking::{DefaultTerminateSignal, EmptySearchStats};
use core::num::NonZeroU8;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Move {
    MoveToFoundation { from: usize, foundation: usize },
    MoveToStack { from: usize, to: usize },
    FlipCard { stack: usize },
    MoveKingToEmpty { from: usize },
}

pub fn move_index(m: &Move) -> usize {
    match m {
        Move::MoveToFoundation { from, foundation } => from * 4 + foundation,
        Move::MoveToStack { from, to } => 28 + from * 7 + to,
        Move::FlipCard { stack } => 77 + stack,
        Move::MoveKingToEmpty { from } => 84 + from,
    }
}

pub fn move_from_index(index: usize) -> Option<Move> {
    match index {
        0..=27 => {
            let from = index / 4;
            let foundation = index % 4;
            Some(Move::MoveToFoundation { from, foundation })
        }
        28..=76 => {
            let relative = index - 28;
            let from = relative / 7;
            let to = relative % 7;
            Some(Move::MoveToStack { from, to })
        }
        77..=83 => Some(Move::FlipCard { stack: index - 77 }),
        84..=90 => Some(Move::MoveKingToEmpty { from: index - 84 }),
        _ => None,
    }
}

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

fn parse_move(data: &str) -> Option<EngineMove> {
    // very small parser expecting format "<TYPE> <idx>"
    let mut parts = data.split_whitespace();
    let t = parts.next()?;
    let idx: u8 = parts.next()?.parse().ok()?;
    Some(match t {
        "DS" => EngineMove::DeckStack(crate::card::Card::from_mask_index(idx)),
        "PS" => EngineMove::PileStack(crate::card::Card::from_mask_index(idx)),
        "DP" => EngineMove::DeckPile(crate::card::Card::from_mask_index(idx)),
        "SP" => EngineMove::StackPile(crate::card::Card::from_mask_index(idx)),
        "R" => EngineMove::Reveal(crate::card::Card::from_mask_index(idx)),
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
    let st = decode_state(encoded).ok_or_else(|| PyValueError::new_err("invalid state"))?;
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
    let v: Value =
        serde_json::from_str(state).map_err(|_| PyValueError::new_err("Invalid state"))?;
    let encoded = v
        .get("encoded")
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
pub fn encode_observation(state: &str) -> PyResult<Vec<f32>> {
    /// Convert a card string like "AH" or "tc" to a 0..51 index.
    fn card_index(s: &str) -> Option<usize> {
        if s.len() < 2 {
            return None;
        }
        let (rank_str, suit_ch) = s.split_at(s.len() - 1);
        let rank = match rank_str.to_ascii_uppercase().as_str() {
            "A" => 0,
            "2" => 1,
            "3" => 2,
            "4" => 3,
            "5" => 4,
            "6" => 5,
            "7" => 6,
            "8" => 7,
            "9" => 8,
            "10" => 9,
            "J" => 10,
            "Q" => 11,
            "K" => 12,
            _ => return None,
        };
        let suit = match suit_ch.to_ascii_uppercase().as_str() {
            "H" => 0,
            "D" => 1,
            "C" => 2,
            "S" => 3,
            _ => return None,
        };
        Some(rank * 4 + suit)
    }

    let v: Value =
        serde_json::from_str(state).map_err(|_| PyValueError::new_err("Invalid JSON state"))?;

    // Three one-hot segments of 52 features each.
    let mut tableau_vec = vec![0.0f32; 52];
    let mut foundation_vec = vec![0.0f32; 52];
    let mut other_vec = vec![0.0f32; 52];

    // Foundations
    if let Some(founds) = v.get("foundations").and_then(|f| f.as_array()) {
        for stack in founds {
            if let Some(cards) = stack.as_array() {
                for card in cards {
                    if let Some(s) = card.as_str() {
                        if let Some(idx) = card_index(s) {
                            foundation_vec[idx] = 1.0;
                        }
                    }
                }
            }
        }
    }

    // Tableau columns
    if let Some(columns) = v.get("tableau").and_then(|t| t.as_array()) {
        for col in columns {
            if let Some(arr) = col.as_array() {
                for card in arr {
                    if let Some(s) = card.as_str() {
                        if let Some(idx) = card_index(s) {
                            if s.chars().last().map_or(false, |c| c.is_lowercase()) {
                                other_vec[idx] = 1.0;
                            } else {
                                tableau_vec[idx] = 1.0;
                            }
                        }
                    }
                }
            } else if let Some(obj) = col.as_object() {
                if let Some(cards_val) = obj.get("cards").and_then(|c| c.as_array()) {
                    let face_down = obj.get("face_down").and_then(|d| d.as_u64()).unwrap_or(0);
                    for (i, card) in cards_val.iter().enumerate() {
                        if let Some(s) = card.as_str() {
                            if let Some(idx) = card_index(s) {
                                if (i as u64) < face_down {
                                    other_vec[idx] = 1.0;
                                } else {
                                    tableau_vec[idx] = 1.0;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Waste pile
    if let Some(waste) = v.get("waste").and_then(|w| w.as_array()) {
        for card in waste {
            if let Some(s) = card.as_str() {
                if let Some(idx) = card_index(s) {
                    other_vec[idx] = 1.0;
                }
            }
        }
    }

    // Stock cards
    if let Some(stock) = v.get("stock").and_then(|w| w.as_array()) {
        for card in stock {
            if let Some(s) = card.as_str() {
                if let Some(idx) = card_index(s) {
                    other_vec[idx] = 1.0;
                }
            }
        }
    }

    let mut obs = Vec::with_capacity(156);
    obs.extend(tableau_vec);
    obs.extend(foundation_vec);
    obs.extend(other_vec);
    Ok(obs)
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
pub fn is_lost(state: &str) -> PyResult<bool> {
    let st = decode_state(state)
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("invalid state"))?;
    let engine: SolitaireEngine<crate::pruning::FullPruner> = st.into();
    let legal_moves = engine.list_moves();
    Ok(legal_moves.is_empty() && !engine.state().is_win())
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

fn foundation_total(v: &Value) -> usize {
    v.get("foundations")
        .and_then(|f| f.as_array())
        .map(|arr| {
            arr.iter()
                .map(|stack| {
                    stack
                        .as_array()
                        .map(|a| a.len())
                        .or_else(|| stack.as_u64().map(|n| n as usize))
                        .unwrap_or(0)
                })
                .sum()
        })
        .unwrap_or(0)
}

fn count_empty_columns(v: &Value) -> usize {
    v.get("tableau")
        .and_then(|t| t.as_array())
        .map(|cols| {
            cols
                .iter()
                .filter(|col| {
                    if let Some(obj) = col.as_object() {
                        let cards = obj
                            .get("cards")
                            .and_then(|c| c.as_array())
                            .map(|a| a.len())
                            .unwrap_or(0);
                        let down = obj
                            .get("face_down")
                            .and_then(|d| d.as_u64())
                            .unwrap_or(0);
                        cards == 0 && down == 0
                    } else if let Some(arr) = col.as_array() {
                        arr.is_empty()
                    } else {
                        false
                    }
                })
                .count()
        })
        .unwrap_or(0)
}

#[pyfunction]
pub fn infer_intention(before: &str, mv: &str, after: &str) -> &'static str {
    let before_v: Value = match serde_json::from_str(before) {
        Ok(v) => v,
        Err(_) => return "",
    };
    let after_v: Value = match serde_json::from_str(after) {
        Ok(v) => v,
        Err(_) => return "",
    };

    if mv.starts_with('R') {
        return "reveal";
    }

    if foundation_total(&after_v) > foundation_total(&before_v) {
        return "foundation";
    }

    let mut parts = mv.split_whitespace();
    let mv_type = parts.next().unwrap_or("");
    let idx: i32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);

    if mv_type == "DP" || mv_type == "SP" {
        if idx / 4 == 12 && count_empty_columns(&before_v) > count_empty_columns(&after_v) {
            return "king_to_empty";
        }
        return "stack_move";
    }
    if mv_type == "DS" || mv_type == "PS" {
        return "foundation";
    }
    "stack_move"
}

#[pyfunction(name = "move_index")]
pub fn move_index_py(_mv: &str) -> PyResult<usize> {
    Ok(0)
}

#[pyfunction(name = "move_from_index")]
pub fn move_from_index_py(_idx: usize) -> PyResult<Option<String>> {
    Ok(None)
}

/// Convert a move string like "DP 10" to a stable numeric index.
#[pyfunction]
pub fn move_to_index(mv: &str) -> PyResult<usize> {
    let mv = parse_move(mv).ok_or_else(|| PyValueError::new_err("invalid move"))?;
    let idx = match mv {
        EngineMove::DeckStack(c) => 0 * 52 + c.mask_index() as usize,
        EngineMove::PileStack(c) => 1 * 52 + c.mask_index() as usize,
        EngineMove::DeckPile(c) => 2 * 52 + c.mask_index() as usize,
        EngineMove::StackPile(c) => 3 * 52 + c.mask_index() as usize,
        EngineMove::Reveal(c) => 4 * 52 + c.mask_index() as usize,
    };
    Ok(idx)
}

/// Reverse of `move_to_index`, recover the move string from its index.
#[pyfunction]
pub fn index_to_move(idx: usize) -> PyResult<String> {
    if idx >= 5 * 52 {
        return Err(PyValueError::new_err("index out of range"));
    }
    let card = crate::card::Card::from_mask_index((idx % 52) as u8);
    let mv = match idx / 52 {
        0 => EngineMove::DeckStack(card),
        1 => EngineMove::PileStack(card),
        2 => EngineMove::DeckPile(card),
        3 => EngineMove::StackPile(card),
        4 => EngineMove::Reveal(card),
        _ => unreachable!(),
    };
    Ok(move_to_string(mv))
}

#[pyfunction]
pub fn solve_klondike(state_json: &str) -> PyResult<Vec<(String, String)>> {
    let v: Value = serde_json::from_str(state_json)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let encoded = v
        .get("encoded")
        .and_then(|e| e.as_str())
        .ok_or_else(|| PyValueError::new_err("missing encoded"))?;

    let mut state = decode_state(encoded)
        .ok_or_else(|| PyValueError::new_err("invalid state"))?;

    let (status, history) =
        solve_with_tracking(&mut state, &EmptySearchStats {}, &DefaultTerminateSignal {});

    if status != SearchResult::Solved {
        return Ok(Vec::new());
    }

    let (hist, _labeled) = history.ok_or_else(|| PyValueError::new_err("no history"))?;

    let mut game = decode_state(encoded).ok_or_else(|| PyValueError::new_err("invalid state"))?;
    let mut before_json = encode_state_to_json(encoded)?;
    let mut res: Vec<(String, String)> = Vec::new();

    for mv in hist.iter() {
        game.do_move(*mv);
        let after_encoded = encode_state(&game);
        let after_json = encode_state_to_json(&after_encoded)?;
        let mv_str = move_to_string(*mv);
        let intent = infer_intention(&before_json, &mv_str, &after_json);
        res.push((mv_str, intent.to_string()));
        before_json = after_json;
    }

    Ok(res)
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
    m.add_function(wrap_pyfunction!(is_lost, m)?)?;
    m.add_function(wrap_pyfunction!(move_index_py, m)?)?;
    m.add_function(wrap_pyfunction!(move_from_index_py, m)?)?;
    m.add_function(wrap_pyfunction!(move_to_index, m)?)?;
    m.add_function(wrap_pyfunction!(index_to_move, m)?)?;
    m.add_function(wrap_pyfunction!(infer_intention, m)?)?;
    m.add_function(wrap_pyfunction!(solve_klondike, m)?)?;
    m.add_function(wrap_pyfunction!(shuffle_seed, m)?)?;
    m.add_function(wrap_pyfunction!(encode_state_to_json, m)?)?;
    m.add_function(wrap_pyfunction!(compute_base_reward_json, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    #[test]
    fn test_new_game_returns_json() {
        let state_json = new_game(None).unwrap();
        let v: Value = serde_json::from_str(&state_json).unwrap();
        assert!(v.get("encoded").is_some());
        assert!(v.get("moves").is_some());
    }

    #[test]
    fn test_legal_moves_matches_moves_field() {
        let state_json = new_game(None).unwrap();
        let v: Value = serde_json::from_str(&state_json).unwrap();
        let encoded = v["encoded"].as_str().unwrap();
        let moves_field: Vec<String> = v["moves"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m.as_str().unwrap().to_string())
            .collect();
        let legal = legal_moves(encoded).unwrap();
        assert_eq!(legal, moves_field);
    }

    #[test]
    fn test_play_move_valid() {
        let state_json = new_game(None).unwrap();
        let v: Value = serde_json::from_str(&state_json).unwrap();
        let mv = v["moves"].as_array().unwrap()[0].as_str().unwrap();
        let (next_state, valid) = play_move(&state_json, mv).unwrap();
        assert!(valid);
        let v_next: Value = serde_json::from_str(&next_state).unwrap();
        assert!(v_next.get("encoded").is_some());
    }

    #[test]
    fn test_compute_base_reward_json() {
        let json = "{\"foundations\": [[\"AH\", \"2H\"], [], [], []]}";
        let r = compute_base_reward_json(json).unwrap();
        assert!((r - 2.0 / 52.0).abs() < 1e-6);
    }

    #[test]
    fn test_encode_state_to_json_roundtrip() {
        let state_json = new_game(None).unwrap();
        let v: Value = serde_json::from_str(&state_json).unwrap();
        let encoded = v["encoded"].as_str().unwrap();
        let out = encode_state_to_json(encoded).unwrap();
        let v_out: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v_out["encoded"].as_str().unwrap(), encoded);
    }
}
