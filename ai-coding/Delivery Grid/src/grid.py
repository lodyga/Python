"""Defines the city grid the delivery robot navigates.

Example:
    g = Grid(5, 5)
    g.set_cell(0, 0, CellType.PICKUP)
    g.set_cell(4, 4, CellType.DEPOT)
    g.set_traffic(2, 2, 5)  # entering (2, 2) costs 5
    g.get_move_cost(1, 2, 2, 2)  # -> 5
"""

from enum import Enum
from typing import List, Tuple, Optional


class CellType(Enum):
    EMPTY = 0
    BLOCKED = 1
    PICKUP = 2
    DEPOT = 3
    TELEPORT_A = 4
    TELEPORT_B = 5


class Grid:
    """Represents the city grid the delivery robot navigates."""

    def __init__(self, rows: int, cols: int):
        self._rows = rows
        self._cols = cols
        self._cells: List[List[CellType]] = [
            [CellType.EMPTY for _ in range(cols)] for _ in range(rows)
        ]
        self._traffic: List[List[int]] = [
            [1 for _ in range(cols)] for _ in range(rows)
        ]
        self._tele_a: Optional[Tuple[int, int]] = None
        self._tele_b: Optional[Tuple[int, int]] = None
        self._teleport: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None

    @property
    def rows(self) -> int:
        return self._rows

    @property
    def cols(self) -> int:
        return self._cols

    def set_cell(self, row: int, col: int, cell_type: CellType) -> None:
        self._cells[row][col] = cell_type
        # todo
        if cell_type == CellType.TELEPORT_A:
            self._tele_a = (row, col)
        elif cell_type == CellType.TELEPORT_B:
            self._tele_b = (row, col)
        elif cell_type in (CellType.BLOCKED, CellType.EMPTY):
            if (row, col) == self._tele_a:
                self._tele_a = None
            elif (row, col) == self._tele_b:
                self._tele_b = None

        if self._tele_a is not None and self._tele_b is not None:
            self._teleport = (self._tele_a, self._tele_b)
        else:
            self._teleport = None

        
    def set_traffic(self, row: int, col: int, cost: int) -> None:
        """Set the movement cost for entering a cell. Must be >= 1."""
        self._traffic[row][col] = cost

    def get_cell(self, row: int, col: int) -> CellType:
        return self._cells[row][col]

    def get_cost(self, row: int, col: int) -> int:
        return self._traffic[row][col]

    def get_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Return walkable neighbors (up, down, left, right)."""
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            
            if 0 <= nr < self._rows and 0 <= nc < self._cols:
                if self._cells[nr][nc] != CellType.BLOCKED:
                    neighbors.append((nr, nc))
        return neighbors

    def get_move_cost(self, from_row: int, from_col: int, to_row: int, to_col: int) -> int:
        """Cost to move from one cell to an adjacent cell."""
        return self._traffic[to_row][to_col]

    def get_teleport_pair(self) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        return self._teleport

    def get_pickups(self) -> List[Tuple[int, int]]:
        result = []
        
        for r in range(self._rows):
            for c in range(self._cols):
                if self._cells[r][c] == CellType.PICKUP:
                    result.append((r, c))
        return result

    def get_depot(self) -> Optional[Tuple[int, int]]:
        for r in range(self._rows):
            for c in range(self._cols):
                if self._cells[r][c] == CellType.DEPOT:
                    return (r, c)
        return None
