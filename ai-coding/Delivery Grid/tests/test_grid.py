"""Tests for Grid."""

import unittest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from grid import Grid, CellType


class TestGridNeighbors(unittest.TestCase):
    def test_center_cell_has_four_neighbors(self) -> None:
        grid = Grid(5, 5)
        neighbors = grid.get_neighbors(2, 2)
        self.assertEqual(len(neighbors), 4)
        self.assertIn((1, 2), neighbors)
        self.assertIn((3, 2), neighbors)
        self.assertIn((2, 1), neighbors)
        self.assertIn((2, 3), neighbors)

    def test_corner_cell_has_two_neighbors(self) -> None:
        grid = Grid(5, 5)
        neighbors = grid.get_neighbors(0, 0)
        self.assertEqual(len(neighbors), 2)
        self.assertIn((1, 0), neighbors)
        self.assertIn((0, 1), neighbors)

    def test_edge_cell_has_three_neighbors(self) -> None:
        grid = Grid(5, 5)
        neighbors = grid.get_neighbors(0, 2)
        self.assertEqual(len(neighbors), 3)

    def test_blocked_neighbor_excluded(self) -> None:
        grid = Grid(5, 5)
        grid.set_cell(2, 3, CellType.BLOCKED)
        neighbors = grid.get_neighbors(2, 2)
        self.assertNotIn((2, 3), neighbors)
        self.assertEqual(len(neighbors), 3)

    def test_bottom_right_corner(self) -> None:
        grid = Grid(5, 5)
        neighbors = grid.get_neighbors(4, 4)
        self.assertEqual(len(neighbors), 2)
        self.assertIn((3, 4), neighbors)
        self.assertIn((4, 3), neighbors)


class TestGridMoveCost(unittest.TestCase):
    def test_default_cost_is_one(self) -> None:
        grid = Grid(5, 5)
        cost = grid.get_move_cost(0, 0, 0, 1)
        self.assertEqual(cost, 1)

    def test_cost_uses_destination(self) -> None:
        grid = Grid(5, 5)
        grid.set_traffic(0, 1, 5)
        cost = grid.get_move_cost(0, 0, 0, 1)
        self.assertEqual(cost, 5)

    def test_cost_not_source(self) -> None:
        grid = Grid(5, 5)
        grid.set_traffic(0, 0, 10)
        cost = grid.get_move_cost(0, 0, 0, 1)
        self.assertEqual(cost, 1)


class TestGridPickupsAndDepot(unittest.TestCase):
    def test_find_pickups(self) -> None:
        grid = Grid(5, 5)
        grid.set_cell(1, 1, CellType.PICKUP)
        grid.set_cell(3, 3, CellType.PICKUP)
        pickups = grid.get_pickups()
        self.assertEqual(len(pickups), 2)
        self.assertIn((1, 1), pickups)
        self.assertIn((3, 3), pickups)

    def test_find_depot(self) -> None:
        grid = Grid(5, 5)
        grid.set_cell(4, 4, CellType.DEPOT)
        depot = grid.get_depot()
        self.assertEqual(depot, (4, 4))

    def test_no_depot_returns_none(self) -> None:
        grid = Grid(5, 5)
        self.assertIsNone(grid.get_depot())


class TestGridTeleport(unittest.TestCase):
    def test_teleport_pair_both_set(self) -> None:
        grid = Grid(5, 5)
        grid.set_cell(0, 0, CellType.TELEPORT_A)
        grid.set_cell(4, 4, CellType.TELEPORT_B)
        pair = grid.get_teleport_pair()
        self.assertIsNotNone(pair)
        self.assertEqual(pair[0], (0, 0))
        self.assertEqual(pair[1], (4, 4))

    def test_teleport_pair_partial(self) -> None:
        grid = Grid(5, 5)
        grid.set_cell(0, 0, CellType.TELEPORT_A)
        self.assertIsNone(grid.get_teleport_pair())

    def test_teleport_cleared_on_overwrite_a(self) -> None:
        # todo
        """Overwriting TELEPORT_A with a non-teleport must invalidate the pair."""
        grid = Grid(5, 5)
        grid.set_cell(0, 0, CellType.TELEPORT_A)
        grid.set_cell(4, 4, CellType.TELEPORT_B)
        self.assertIsNotNone(grid.get_teleport_pair())
        grid.set_cell(0, 0, CellType.EMPTY)
        self.assertIsNone(grid.get_teleport_pair())

    def test_teleport_cleared_on_overwrite_b(self) -> None:
        """Overwriting TELEPORT_B with a non-teleport must invalidate the pair."""
        grid = Grid(5, 5)
        grid.set_cell(0, 0, CellType.TELEPORT_A)
        grid.set_cell(4, 4, CellType.TELEPORT_B)
        self.assertIsNotNone(grid.get_teleport_pair())
        grid.set_cell(4, 4, CellType.BLOCKED)
        self.assertIsNone(grid.get_teleport_pair())


if __name__ == "__main__":
    unittest.main()
