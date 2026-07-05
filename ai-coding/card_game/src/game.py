"""You'll implement this. Game logic for the Triplet Draw card problem."""

from typing import List, Tuple
from hand import Hand


class Game:
    """
    Given a hand of cards, find triples whose scoring values sum to a target.

    Triples are unordered (i < j < k). Duplicate index triples must not appear.
    Two triples with the same indices in different order count as one.
    """

    def __init__(self, hand: Hand):
        self.hand = hand

    def validate_request(self, target: int) -> None:
        """
        Reject clearly-invalid requests.
          - target must be a positive integer (>= 3, minimum possible with three 1-value cards)
          - hand must have at least 3 cards
        Raise ValueError otherwise.
        """
        if target < 3:
            raise ValueError(f"target must be >= 3, got {target}")
        if self.hand.size() < 3:
            raise ValueError(
                f"hand must have at least 3 cards, has {self.hand.size()}"
            )

    def find_triplets(self, target: int) -> List[Tuple[int, int, int]]:
        """
        Return every (i, j, k) with i < j < k such that
            hand.card_value(i) + hand.card_value(j) + hand.card_value(k) == target

        Each triple appears exactly once. The returned list must be sorted
        lexicographically so tests can compare deterministically.

        Call self.validate_request(target) first to reject invalid input
        (target < 3 or hand size < 3) with a ValueError.
        """
        self.validate_request(target)
        res = []

        hand = self.hand
        hand_size = hand.size()
        
        val_to_idx = [(hand.card_value(idx), idx) for idx in range(hand_size)]
        val_to_idx.sort()

        for i in range(hand.size() - 2):
            vi, idx_i = val_to_idx[i]
            j = i + 1
            k = hand.size() - 1

            while j < k:
                vj, idx_j = val_to_idx[j]
                vk, idx_k = val_to_idx[k]

                triplet = vi + vj + vk

                if triplet == target:
                    tmp_res = [(idx_i, idx_j, idx_k)]
                    org_k = k
                    while j < k - 1 and val_to_idx[k - 1][0] == vk:
                        tmp_res.append((idx_i, idx_j, val_to_idx[k - 1][1]))
                        k -= 1
                    tmp_res.reverse()
                    res.extend(tmp_res)
                    k = org_k
                    j += 1
                elif triplet < target:
                    j += 1
                else:
                    k -= 1

        return res
