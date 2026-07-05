"""You'll implement this."""

from typing import List, Tuple
from grid import Grid


class DeliverySolver:
    """Finds the minimum-cost route visiting all pickups then the depot."""

    def __init__(self, grid: Grid, start: Tuple[int, int]):
        self.grid = grid
        self.start = start
        self.route: List[Tuple[int, int]] = []
        self.total_cost = 0

    def find_route(self) -> List[Tuple[int, int]]:
        """
        Find minimum-cost route from start, visiting all pickups, ending at depot.

        Rules:
        - Move to adjacent cells (up/down/left/right) paying the destination traffic cost.
        - Cannot enter BLOCKED cells.
        - Must visit every PICKUP cell at least once.
        - Must end at the DEPOT cell.
        - May use the teleport pair once (zero cost, instant move between the two pads).

        Returns:
            List of (row, col) representing the path from start to depot.
            Empty list if no valid route exists.
        """
        pass

    def get_route(self) -> List[Tuple[int, int]]:
        return self.route

    def get_total_cost(self) -> int:
        return self.total_cost
