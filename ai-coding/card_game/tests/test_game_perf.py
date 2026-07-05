"""Performance tests for Game.find_triplets. Phase 3 optimization gate.

Naive O(n^3) works for small hands but blows up past n ~ 500.
An O(n^2) solution using a running two-pointer or a complement hash map
should comfortably finish every test below.

Hand construction is deliberately SPARSE: most cards share a single rank
that cannot contribute to the target, with a handful of "solver" cards
planted that form O(n) triples with the background cards. This tests
the iteration cost (n^2) without blowing up the result list size, so
a 1200-card hand fits comfortably in the sandbox memory budget while
still timing out any O(n^3) solution.
"""

import unittest
import sys
import os
import time
from typing import List
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from card import Card, Suit
from hand import Hand
from game import Game


def _sparse_hand(n: int, *, background_rank: int, solver_ranks: List[int]) -> Hand:
    """Build a sparse hand: n cards total, most of background_rank, a few solver_ranks.

    Solver cards are planted last so triples are (bg_i, bg_j, solver) or similar,
    which keeps the number of valid triples ~O(n) rather than ~O(n^3).
    """
    hand = Hand(ace_high=False)
    suits = [Suit.SPADES, Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS]
    bg_count = n - len(solver_ranks)
    for i in range(bg_count):
        hand.add(Card(background_rank, suits[i % 4]))
    for j, rank in enumerate(solver_ranks):
        hand.add(Card(rank, suits[j % 4]))
    return hand


class TestFindTripletsPerformance(unittest.TestCase):
    def test_medium_hand_correctness(self) -> None:
        """900 cards: 897 rank-2 + rank-5 + rank-3 + rank-10, target=15.

        Valid triples: rank-10 (value 10) + rank-3 (value 3) + any rank-2 (value 2)
        = 897 triples. Background rank-2 never sums to 15 with itself (2+2+2=6),
        so the result list stays ~O(n) while the O(n^2) iteration runs fully.
        Naive O(n^3) ~ 1.2e8 ops comfortably overshoots the 3s time budget in
        Python; optimized finishes in ~0.05s.
        """
        hand = _sparse_hand(900, background_rank=2, solver_ranks=[5, 3, 10])
        game = Game(hand)
        start = time.time()
        result = game.find_triplets(15)
        elapsed = time.time() - start
        self.assertIsInstance(result, list)
        self.assertGreater(
            len(result), 0, "optimized solution must still return triples"
        )
        self.assertLess(
            elapsed,
            3.0,
            f"Took {elapsed:.2f}s on n=900, must be under 3s (use O(n^2))",
        )

    def test_large_hand_stays_under_limit(self) -> None:
        """1500 cards: same sparse pattern, scaled up to verify O(n^2) iteration cost.

        Optimized solution runs ~2.25M complement lookups; naive O(n^3) ~ 5.6e8 ops
        times out hard in Python. Result-list size stays ~O(n) (1497 triples), so
        there's no risk of sandbox OOM - which was why the old dense n=1200 hand
        used to need a results cap inside game.py.
        """
        hand = _sparse_hand(1500, background_rank=2, solver_ranks=[5, 3, 10])
        game = Game(hand)
        start = time.time()
        result = game.find_triplets(15)
        elapsed = time.time() - start
        self.assertIsInstance(result, list)
        self.assertGreater(
            len(result), 0, "optimized solution must still return triples"
        )
        self.assertLess(
            elapsed,
            6.0,
            f"Took {elapsed:.2f}s on n=1500, must be under 6s (O(n^3) will NOT pass)",
        )


if __name__ == "__main__":
    unittest.main()
