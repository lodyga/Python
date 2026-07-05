"""Tests for Game.find_triplets. Phase 2."""

import unittest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from card import Card, Suit
from hand import Hand
from game import Game


def _hand_from(values):
    """Build a hand where each input value (1..10) becomes a card worth that value.

    Only values in 1..10 are supported, which is sufficient for every test in
    this file. ace_high is False so ace (rank 1) scores as 1.
    """
    hand = Hand(ace_high=False)
    for v in values:
        if not (1 <= v <= 10):
            raise ValueError(f"_hand_from expects values in 1..10, got {v}")
        if v == 1:
            hand.add(Card(1, Suit.SPADES))
        else:
            hand.add(Card(v, Suit.HEARTS))
    return hand


class TestFindTripletsBasic(unittest.TestCase):
    def test_single_triple_sum_15(self) -> None:
        hand = _hand_from([3, 5, 7])
        game = Game(hand)
        result = game.find_triplets(15)
        self.assertEqual(result, [(0, 1, 2)])

    def test_no_triple_exists(self) -> None:
        hand = _hand_from([1, 2, 3])
        game = Game(hand)
        self.assertEqual(game.find_triplets(15), [])

    def test_multiple_triples(self) -> None:
        """Hand [2,3,4,5,6], target=12 -> (0,2,4)=2+4+6 and (1,2,3)=3+4+5."""
        hand = _hand_from([2, 3, 4, 5, 6])
        game = Game(hand)
        result = game.find_triplets(12)
        self.assertEqual(result, [(0, 2, 4), (1, 2, 3)])

    def test_result_is_sorted_lex(self) -> None:
        hand = _hand_from([5, 5, 5, 5])
        # [4, 5, 5, 5, 5, 6, 6, 6, 6]
        game = Game(hand)
        result = game.find_triplets(15)
        self.assertEqual(
            result,
            [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)],
        )

    def test_face_cards_all_score_ten(self) -> None:
        """Three Jacks should sum to 30 (10+10+10), not 33."""
        hand = Hand()
        hand.add(Card(11, Suit.SPADES))
        hand.add(Card(12, Suit.HEARTS))
        hand.add(Card(13, Suit.DIAMONDS))
        game = Game(hand)
        self.assertEqual(game.find_triplets(30), [(0, 1, 2)])
        self.assertEqual(game.find_triplets(33), [])


class TestFindTripletsEdges(unittest.TestCase):
    def test_less_than_three_cards_rejects(self) -> None:
        hand = _hand_from([5, 10])
        game = Game(hand)
        with self.assertRaises(ValueError):
            game.find_triplets(15)

    def test_invalid_target_rejects(self) -> None:
        hand = _hand_from([5, 5, 5])
        game = Game(hand)
        with self.assertRaises(ValueError):
            game.find_triplets(2)

    def test_dupes_same_rank_different_suit(self) -> None:
        """Two 5-of-hearts-like duplicates with different suits are distinct cards."""
        hand = Hand()
        hand.add(Card(5, Suit.SPADES))
        hand.add(Card(5, Suit.HEARTS))
        hand.add(Card(5, Suit.DIAMONDS))
        game = Game(hand)
        self.assertEqual(game.find_triplets(15), [(0, 1, 2)])


if __name__ == "__main__":
    unittest.main()
