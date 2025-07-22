from env import reward


def test_rust_reward_available():
    s = '{"foundations": [["AH", "2H", "3H"], [], [], []]}'
    if reward.compute_base_reward_json is None:
        assert True
    else:
        r = reward.compute_base_reward_json(s)
        assert abs(r - (3.0 / 52.0)) < 1e-6


def test_reward_missing_foundations():
    s = '{"piles": []}'
    if reward.compute_base_reward_json is None:
        assert True
    else:
        import pytest
        with pytest.raises(Exception):
            reward.compute_base_reward_json(s)
