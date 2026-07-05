"""The player's hand of cards. Computes scoring values from a card's rank."""

from typing import List
from card import Card


class Hand:
    """A collection of cards with rules for how each card scores."""

    def __init__(self, ace_high: bool = False):
        """
        ace_high:
            False -> Ace scores as 1 (default, used for low-sum games like target=15)
            True  -> Ace scores as 11 (used for games like blackjack variants)
        """
        self._cards: List[Card] = []
        self._ace_high = ace_high

    def add(self, card: Card) -> None:
        self._cards.append(card)

    def size(self) -> int:
        return len(self._cards)

    def get(self, index: int) -> Card:
        return self._cards[index]

    def all_cards(self) -> List[Card]:
        return list(self._cards)

    def card_value(self, index: int) -> int:
        """
        Scoring value for the card at index. Rules:
          - Ace (rank 1): 11 if ace_high else 1
          - Face cards (Jack=11, Queen=12, King=13): all score as 10
          - All others (2-10): score as their rank
        """
        card = self._cards[index]
        rank = card.rank
        
        if rank == 1:
            return 1 + (10 if self._ace_high else 0)
        elif rank < 11:
            return rank
        else:
            return 10
