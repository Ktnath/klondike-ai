import klondike_core as kc
import pytest

HAS_CORE = kc._core is not None


def test_shuffle_seed_random(monkeypatch):
    if HAS_CORE:
        monkeypatch.delenv("SHUFFLE_SEED", raising=False)
        first = kc.shuffle_seed()
        second = kc.shuffle_seed()
        assert first != second
    else:
        with pytest.raises(NotImplementedError):
            kc.shuffle_seed()
