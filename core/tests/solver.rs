use serde_json::Value;
use klondike_core::{solve_klondike, new_game, move_from_index, move_index, Move};

#[test]
fn test_solver_returns_json() {
    let state = new_game(None).unwrap();
    let out = solve_klondike(&state).unwrap();
    let v: Value = serde_json::from_str(&out).unwrap();
    assert!(v.is_array() || v.is_null());
}

#[test]
fn test_move_index_bidirectional_mapping() {
    // Validate that every valid index produces a Move that maps back to the same index
    for index in 0..91 {
        let m = move_from_index(index)
            .unwrap_or_else(|| panic!("move_from_index({}) returned None", index));
        let recovered_index = move_index(&m);
        assert_eq!(
            index, recovered_index,
            "Index mismatch: original {}, recovered {} for move {:?}",
            index, recovered_index, m
        );
    }
}

#[test]
fn test_invalid_indices_return_none() {
    // Ensure indices outside the valid range return None
    for invalid_index in [91, 100, 255, 999] {
        assert!(move_from_index(invalid_index).is_none());
    }
}
