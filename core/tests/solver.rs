use serde_json::Value;
use klondike_core::{solve_klondike, new_game, move_from_index, move_index};

#[test]
fn test_solver_returns_json() {
    let state = new_game(None).unwrap();
    let out = solve_klondike(&state).unwrap();
    let v: Value = serde_json::from_str(&out).unwrap();
    assert!(v.is_array() || v.is_null());
}

#[test]
fn test_move_index_bidirectionality() {
    for i in 0..91 {
        let m = move_from_index(i).unwrap();
        assert_eq!(move_index(&m), i);
    }
}
