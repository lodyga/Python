"""Tests for Hand."""

import unittest
import sys
import os
import random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from card import Card, Suit
from hand import Hand


class TestHandBasic(unittest.TestCase):
    def test_empty_hand(self) -> None:
        hand = Hand()
        self.assertEqual(hand.size(), 0)

    def test_add_and_size(self) -> None:
        rank_a, rank_b = random.sample(range(2, 11), 2)
        suit_a, suit_b = random.sample(tuple(Suit), 2)
        hand = Hand()
        hand.add(Card(rank_a, suit_a))
        hand.add(Card(rank_b, suit_b))
        self.assertEqual(hand.size(), 2)

    def test_get_by_index(self) -> None:
        rank = random.choice(range(2, 11))
        suit = random.choice(tuple(Suit))
        hand = Hand()
        hand.add(Card(rank, suit))
        card = hand.get(0)
        self.assertEqual(card.rank, rank)
        self.assertEqual(card.suit, suit)


class TestHandCardValue(unittest.TestCase):
    def test_number_card_scores_as_rank(self) -> None:
        """2-10 score as their own rank."""
        rank_a, rank_b, rank_c = random.choices(range(2, 11), k=3)
        suit_a, suit_b, suit_c = random.choices(tuple(Suit), k=3)
        hand = Hand()
        hand.add(Card(rank_a, suit_a))
        hand.add(Card(rank_b, suit_b))
        hand.add(Card(rank_c, suit_c))
        self.assertEqual(hand.card_value(0), rank_a)
        self.assertEqual(hand.card_value(1), rank_b)
        self.assertEqual(hand.card_value(2), rank_c)

    def test_face_cards_all_score_as_ten(self) -> None:
        """Jack, Queen, King all score as 10, regardless of rank (11, 12, 13)."""
        hand = Hand()
        hand.add(Card(11, Suit.SPADES))   # Jack
        hand.add(Card(12, Suit.HEARTS))   # Queen
        hand.add(Card(13, Suit.DIAMONDS)) # King
        self.assertEqual(hand.card_value(0), 10)
        self.assertEqual(hand.card_value(1), 10)
        self.assertEqual(hand.card_value(2), 10)

    def test_ace_low_scores_as_one(self) -> None:
        """Default (ace_high=False) -> Ace = 1."""
        hand = Hand(ace_high=False)
        hand.add(Card(1, Suit.SPADES))
        self.assertEqual(hand.card_value(0), 1)

    def test_ace_high_scores_as_eleven(self) -> None:
        """When ace_high=True, Ace must score as 11."""
        hand = Hand(ace_high=True)
        hand.add(Card(1, Suit.HEARTS))
        self.assertEqual(hand.card_value(0), 11)


if __name__ == "__main__":
    unittest.main()
