import os
import sys
import json
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from env.is_critical_move import is_critical_move


def test_move_to_foundation():
    before = json.dumps({
        "foundations": [["AH", "2H", "3H"], [], [], []],
        "tableau": [],
        "waste": [],
        "stock": []
    })
    after = json.dumps({
        "foundations": [["AH", "2H", "3H", "4H"], [], [], []],
        "tableau": [],
        "waste": [],
        "stock": []
    })
    assert is_critical_move(before, after) == 0.5


def test_reveal_card():
    before = json.dumps({
        "foundations": [[], [], [], []],
        "tableau": [{"cards": ["7S"], "face_down": 1}],
        "waste": [],
        "stock": []
    })
    after = json.dumps({
        "foundations": [[], [], [], []],
        "tableau": [{"cards": ["7S"], "face_down": 0}],
        "waste": [],
        "stock": []
    })
    assert is_critical_move(before, after) == 1.0


def test_king_to_empty_column():
    before = json.dumps({
        "foundations": [[], [], [], []],
        "tableau": [
            {"cards": [], "face_down": 0},
            {"cards": ["QH"], "face_down": 0}
        ],
        "waste": [],
        "stock": []
    })
    after = json.dumps({
        "foundations": [[], [], [], []],
        "tableau": [
            {"cards": ["KD"], "face_down": 0},
            {"cards": ["QH"], "face_down": 0}
        ],
        "waste": [],
        "stock": []
    })
    assert is_critical_move(before, after) == 0.7


def test_empty_column():
    before = json.dumps({
        "foundations": [[], [], [], []],
        "tableau": [
            {"cards": ["9C"], "face_down": 0},
            {"cards": ["8S"], "face_down": 0}
        ],
        "waste": [],
        "stock": []
    })
    after = json.dumps({
        "foundations": [[], [], [], []],
        "tableau": [
            {"cards": [], "face_down": 0},
            {"cards": ["8S"], "face_down": 0}
        ],
        "waste": [],
        "stock": []
    })
    assert is_critical_move(before, after) == 0.5


def test_game_won():
    before = json.dumps({
        "foundations": [[], [], [], []],
        "tableau": [],
        "waste": [],
        "stock": []
    })
    after = json.dumps({
        "foundations": [[], [], [], []],
        "tableau": [],
        "waste": [],
        "stock": [],
        "is_won": True
    })
    assert is_critical_move(before, after) == 10.0
