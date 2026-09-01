"""Unit tests for the ingest transforms, on a minimal synthetic replay that
mirrors the real Kaggle episode structure (see docs in the source repo)."""
import pytest

from pipeline.ingest import DECK_SIZE, extract_game, to_rows


def make_replay(rewards=(-1, 1), statuses=("DONE", "DONE"), deck_sizes=(60, 60)):
    decks = [[100 + i] * n for i, n in enumerate(deck_sizes)]
    step0 = [
        {"observation": {"current": None, "logs": []}, "visualize": [{"action": decks}]},
        {"observation": {}},
    ]
    # engine reports firstPlayer=-1 until the opening coin flip resolves
    step_undecided = [
        {"observation": {"current": {"firstPlayer": -1}, "logs": []}},
        {"observation": {}},
    ]
    step1 = [
        {
            "observation": {
                "current": {"firstPlayer": 1},
                "logs": [
                    {"cardId": 100, "fromArea": 1, "playerIndex": 0, "serial": 3, "toArea": 2, "type": 6},
                    {"playerIndex": 0, "type": 0},
                ],
            }
        },
        {"observation": {}},
    ]
    return {
        "info": {"EpisodeId": "12345", "TeamNames": ["alice", "bob"]},
        "rewards": list(rewards),
        "statuses": list(statuses),
        "steps": [step0, step_undecided, step1],
    }


def test_extract_happy_path():
    g = extract_game(make_replay(), batch="corpus")
    assert g["episode_id"] == 12345
    assert g["first_player"] == 1
    assert g["n_steps"] == 3
    assert [len(d) for d in g["decks"]] == [DECK_SIZE, DECK_SIZE]
    # events flattened with position, sparse fields None
    assert len(g["events"]) == 2
    assert g["events"][0]["card_id"] == 100
    assert g["events"][1]["card_id"] is None


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"rewards": (0, 0)}, "winner"),          # tie / errored game
        ({"statuses": ("DONE", "TIMEOUT")}, "DONE"),
        ({"deck_sizes": (59, 60)}, "decklists"),
    ],
)
def test_extract_rejects_contract_violations(kwargs, match):
    with pytest.raises(ValueError, match=match):
        extract_game(make_replay(**kwargs), batch="corpus")


def test_to_rows_grain_and_winner():
    g = extract_game(make_replay(), batch="corpus")
    game_row, seat_rows, event_rows = to_rows(g, meta=None)

    assert len(seat_rows) == 2  # fact grain: one row per (game, seat)
    winners = [r["is_winner"] for r in seat_rows]
    assert winners == [False, True]
    assert [r["went_first"] for r in seat_rows] == [False, True]
    # no metadata -> unknown partition, never a fabricated date
    assert game_row["play_date"] == "unknown"
    assert all(r["play_date"] == "unknown" for r in seat_rows + event_rows)


def test_to_rows_joins_metadata():
    from datetime import datetime, timezone

    g = extract_game(make_replay(), batch="corpus")
    meta = {
        "played_at": datetime(2026, 7, 3, 6, 8, 10, tzinfo=timezone.utc),
        "ended_at": datetime(2026, 7, 3, 6, 10, 16, tzinfo=timezone.utc),
        "submission_id": [111, 222],
        "team_id": [16376649, 16393241],
        "rating_before": [1019.9, 1080.7],
        "rating_after": [1016.4, 1084.3],
    }
    game_row, seat_rows, _ = to_rows(g, meta)
    assert game_row["play_date"] == "2026-07-03"
    assert seat_rows[0]["submission_id"] == 111
    assert seat_rows[1]["rating_after"] == 1084.3
