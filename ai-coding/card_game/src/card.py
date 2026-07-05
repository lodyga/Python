"""Represents a single playing card with a rank and suit.

Example:
    c = Card(rank=7, suit=Suit.HEARTS)
    c.rank                            # -> 7
    Card(rank=1, suit=Suit.SPADES)    # Ace of spades (scoring handled by Hand)
"""

from enum import Enum


class Suit(Enum):
    SPADES = "S"
    HEARTS = "H"
    DIAMONDS = "D"
    CLUBS = "C"


class Card:
    """A playing card. Rank is 1-13 where 1=Ace, 11=Jack, 12=Queen, 13=King."""

    def __init__(self, rank: int, suit: Suit):
        if rank < 1 or rank > 13:
            raise ValueError(f"rank must be 1..13, got {rank}")
        self._rank = rank
        self._suit = suit

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def suit(self) -> Suit:
        return self._suit

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Card):
            return False
        
        return (
            self._rank == other._rank and 
            self._suit == other._suit
        )

    def __hash__(self) -> int:
        return hash(self._rank)

    def __repr__(self) -> str:
        names = {1: "A", 11: "J", 12: "Q", 13: "K"}
        label = names.get(self._rank, str(self._rank))
        return f"{label}{self._suit.value}"
