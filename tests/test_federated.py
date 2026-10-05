import chess
import pytest

from federated import aggregate, evaluate, federated_train, train
from q_learning import (
    DEFAULT_FEN_BLACK,
    DEFAULT_FEN_WHITE,
    load_checkpoint,
    save_checkpoint,
)


class TestAggregate:
    def test_means_values(self):
        tables = [
            {"fen1": {10: 3.0, 11: 1.0}},
            {"fen1": {10: 5.0}},
            {"fen1": {10: 7.0, 12: 2.0}},
        ]
        merged = aggregate(tables)
        assert merged["fen1"][10] == pytest.approx(5.0)
        assert merged["fen1"][11] == pytest.approx(1.0)
        assert merged["fen1"][12] == pytest.approx(2.0)

    def test_mean_over_clients_that_saw_the_state(self):
        merged = aggregate([{"fen1": {10: 4.0}}, {}, {"fen1": {10: 8.0}}])
        assert merged["fen1"][10] == pytest.approx(6.0)

    def test_empty(self):
        assert aggregate([{}, {}]) == {}


class TestTrain:
    def test_seeded_runs_are_reproducible(self):
        q_a, q_b = {}, {}
        train(q_a, chess.WHITE, DEFAULT_FEN_WHITE, 3, seed=42)
        train(q_b, chess.WHITE, DEFAULT_FEN_WHITE, 3, seed=42)
        assert q_a == q_b

    def test_different_seeds_differ(self):
        q_a, q_b = {}, {}
        train(q_a, chess.WHITE, DEFAULT_FEN_WHITE, 5, seed=42)
        train(q_b, chess.WHITE, DEFAULT_FEN_WHITE, 5, seed=43)
        assert q_a != q_b


class TestEvaluate:
    def test_black_agent_evaluated_as_black(self):
        # Regression: evaluate() used to put the agent in the white slot,
        # so black agents played the wrong color and scored ~0.
        q = {}
        train(q, chess.BLACK, DEFAULT_FEN_BLACK, 150, seed=5)
        stats = evaluate(q, chess.BLACK, DEFAULT_FEN_BLACK, 10)
        assert sum(stats.values()) == 10
        assert stats["win"] >= 5


class TestFederatedTrain:
    def test_end_to_end_tiny(self, tmp_path):
        q = federated_train(
            chess.WHITE, episodes=2, clients=3, group_size=3, seed_base=7
        )
        assert len(q) > 0
        save_checkpoint(tmp_path / "fed.pkl", q, chess.WHITE, DEFAULT_FEN_WHITE)
        ckpt = load_checkpoint(tmp_path / "fed.pkl")
        assert ckpt["q"] == q
        assert ckpt["color"] == chess.WHITE
