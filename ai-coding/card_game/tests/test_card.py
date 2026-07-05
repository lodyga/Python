"""Tests for Card."""

import unittest
import sys
import os
import random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from card import Card, Suit


class TestCardBasic(unittest.TestCase):
    def test_rank_stored(self) -> None:
        rank = random.choice(range(2, 11))
        card = Card(rank, Suit.SPADES)
        self.assertEqual(card.rank, rank)
        self.assertEqual(card.suit, Suit.SPADES)

    def test_rank_validation_low(self) -> None:
        with self.assertRaises(ValueError):
            Card(0, Suit.HEARTS)

    def test_rank_validation_high(self) -> None:
        with self.assertRaises(ValueError):
            Card(14, Suit.CLUBS)

    def test_repr_uses_suit_letter(self) -> None:
        self.assertEqual(repr(Card(5, Suit.SPADES)), "5S")
        self.assertEqual(repr(Card(11, Suit.HEARTS)), "JH")
        self.assertEqual(repr(Card(1, Suit.DIAMONDS)), "AD")


class TestCardEquality(unittest.TestCase):
    def test_same_rank_same_suit_equal(self) -> None:
        rank = random.choice(range(2, 11))
        suit = random.choice(tuple(Suit))
        a = Card(rank, suit)
        b = Card(rank, suit)
        self.assertEqual(a, b)

    def test_different_suit_not_equal(self) -> None:
        """Two cards with the same rank but different suits must not be equal."""
        rank = random.choice(range(2, 11))
        suit_a, suit_b = random.sample(tuple(Suit), 2)
        a = Card(rank, suit_a)
        b = Card(rank, suit_b)
        self.assertNotEqual(a, b)

    def test_different_rank_not_equal(self) -> None:
        rank_a, rank_b = random.sample(range(2, 11), 2)
        suit = random.choice(tuple(Suit))
        a = Card(rank_a, suit)
        b = Card(rank_b, suit)
        self.assertNotEqual(a, b)


if __name__ == "__main__":
    unittest.main()
