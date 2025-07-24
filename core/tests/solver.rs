use serde_json::Value;
use klondike_core::{solve_klondike, new_game};

#[test]
fn test_solver_returns_json() {
    let state = new_game(None).unwrap();
    let out = solve_klondike(&state).unwrap();
    let v: Value = serde_json::from_str(&out).unwrap();
    assert!(v.is_array() || v.is_null());
}
