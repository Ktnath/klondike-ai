use serde_json::Value;
use klondike_core::solve_klondike;

#[test]
fn test_solver_returns_json() {
    let out = solve_klondike("42").unwrap();
    let v: Value = serde_json::from_str(&out).unwrap();
    assert!(v.get("result").is_some());
}
