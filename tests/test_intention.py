import json
import klondike_core as kc

def test_infer_intention_simple():
    state = kc.new_game()
    encoded = json.loads(state)["encoded"]
    move = kc.legal_moves(encoded)[0]
    intention = kc.infer_intention(state, move)
    assert isinstance(intention, str)
