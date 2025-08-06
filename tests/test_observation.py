def test_encode_observation():
    import klondike_core

    state = klondike_core.new_game()
    obs = klondike_core.encode_observation(state)
    assert isinstance(obs, list)
    assert len(obs) == 156
