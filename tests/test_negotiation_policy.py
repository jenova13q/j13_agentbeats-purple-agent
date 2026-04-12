from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent import Agent  # noqa: E402


def make_agent() -> Agent:
    return Agent()


def test_choose_negotiation_mode_prefers_close_safe_for_late_tough():
    agent = make_agent()
    game_index = 7
    valuations = [9, 4, 1]
    max_value = 80
    agent.state.incoming_offers_to_self_by_game[game_index] = [
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 1],
    ]

    style = agent._infer_opponent_style(game_index, valuations, max_value)
    mode = agent._choose_negotiation_mode(
        game_index=game_index,
        round_index=4,
        max_rounds=5,
        opponent_style=style,
        best_incoming_value=18,
        max_value=max_value,
    )

    assert style == "tough"
    assert mode == "close_safe"
    assert agent.state.negotiation_mode_by_game[game_index] == "close_safe"


def test_accept_uses_late_round_lookahead_to_close():
    agent = make_agent()
    obs = {
        "action": "ACCEPT_OR_REJECT",
        "valuations_self": [10, 1, 1],
        "batna_self": 15,
        "round_index": 5,
        "max_rounds": 5,
        "discount": 0.98,
        "game_index": 3,
        "quantities": [2, 1, 0],
        "pending_offer": {"offer_allocation_self": [2, 0, 0]},
    }

    result = agent._accept_or_walk(obs)

    assert result == {"action": "ACCEPT"}


def test_value_max_mode_for_flexible_opponent_early():
    agent = make_agent()
    game_index = 11
    valuations = [8, 5, 1]
    max_value = 70
    agent.state.incoming_offers_to_self_by_game[game_index] = [
        [0, 0, 0],
        [1, 0, 0],
        [2, 1, 0],
        [0, 1, 1],
    ]

    style = agent._infer_opponent_style(game_index, valuations, max_value)
    mode = agent._choose_negotiation_mode(
        game_index=game_index,
        round_index=2,
        max_rounds=5,
        opponent_style=style,
        best_incoming_value=22,
        max_value=max_value,
    )

    assert style == "flexible"
    assert mode == "value_max"
