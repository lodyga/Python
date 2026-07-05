"""Run this to see how your code is called."""

from card import Card, Suit
from hand import Hand
from game import Game


def main() -> None:
    hand = Hand(ace_high=False)
    hand.add(Card(5, Suit.SPADES))
    hand.add(Card(3, Suit.HEARTS))
    hand.add(Card(7, Suit.DIAMONDS))
    hand.add(Card(11, Suit.CLUBS))   # Jack -> scores as 10
    hand.add(Card(1, Suit.SPADES))   # Ace  -> scores as 1
    hand.add(Card(2, Suit.HEARTS))

    game = Game(hand)
    target = 15
    triples = game.find_triplets(target)

    if triples is None:
        print("find_triplets not implemented yet.")
        return

    print(f"Target sum: {target}")
    print(f"Found {len(triples)} triples:")
    for i, j, k in triples:
        ci, cj, ck = hand.get(i), hand.get(j), hand.get(k)
        vi, vj, vk = hand.card_value(i), hand.card_value(j), hand.card_value(k)
        print(f"  ({i},{j},{k})  {ci}+{cj}+{ck}  =  {vi}+{vj}+{vk} = {vi+vj+vk}")


if __name__ == "__main__":
    main()
