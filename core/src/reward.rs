use crate::GameState;

/// Compute a sparse reward for the given game state.
///
/// Returns +1.0 if the game is won, 0.0 otherwise.
pub fn compute_base_reward(game_state: &GameState) -> f32 {
    if game_state.is_won() {
        1.0
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
    fn test_reward_winning_state() {
        let state = winning_state();
        assert_eq!(compute_base_reward(&state), 1.0);
    }
}
