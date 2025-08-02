import audit_solve_klondike as audit

def test_audit_warns_on_invalid_move_index(monkeypatch, capsys):
    monkeypatch.setattr(audit, 'N', 1)
    monkeypatch.setattr(audit, 'new_game', lambda: '{}')
    # return list of (state, move, intention)
    def fake_solve(state):
        return [({}, 'bad_move', 'intent')]
    monkeypatch.setattr(audit, 'solve_klondike', fake_solve)
    monkeypatch.setattr(audit, 'encode_observation', lambda state: [0] * 100)
    monkeypatch.setattr(audit, 'move_index', lambda mv: -1)

    audit.audit_solve_klondike()
    out = capsys.readouterr().out
    assert 'move_index invalide' in out
