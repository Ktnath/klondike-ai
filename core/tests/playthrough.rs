extern crate klondike_core;
use klondike_core::{GameState, Engine, Move};
use serde_json::json;

fn winning_state() -> GameState {
    let card = json!({"rank": 0u8, "suit": 0u8, "is_face_up": true});
    let foundation: Vec<_> = (0..13).map(|_| card.clone()).collect();
    let foundations = vec![
        serde_json::Value::Array(foundation.clone()),
        serde_json::Value::Array(foundation.clone()),
        serde_json::Value::Array(foundation.clone()),
        serde_json::Value::Array(foundation.clone()),
    ];
    let tableau = vec![
        serde_json::Value::Array(vec![]),
        serde_json::Value::Array(vec![]),
        serde_json::Value::Array(vec![]),
        serde_json::Value::Array(vec![]),
        serde_json::Value::Array(vec![]),
        serde_json::Value::Array(vec![]),
        serde_json::Value::Array(vec![]),
    ];
    let state_json = json!({
        "stock": [],
        "waste": [],
        "tableau": tableau,
        "foundations": foundations,
        "score": 0,
    });
    serde_json::from_value(state_json).unwrap()
}

#[test]
fn test_playthrough() {
    // 1. Create a new game state and engine
    let state = GameState::new();
    let mut engine = Engine::from_state(state);

    // 2. Get legal moves
    let moves = engine.get_available_moves().to_vec();
    assert!(!moves.is_empty());

    // 3. Apply a move (draw a card)
    assert!(engine.make_move(&Move::DrawCard));

    // 4. Verify the waste now has a card
    assert!(engine.get_state().get_top_waste().is_some());

    // 5. Force a winning state and verify detection
    let win_state = winning_state();
    assert!(win_state.is_won());
}
