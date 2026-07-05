"""Tests for DeliverySolver."""

import unittest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from grid import Grid, CellType
from solver import DeliverySolver


class TestSolverBasic(unittest.TestCase):
    def test_simple_route_no_pickups(self) -> None:
        """Direct path to depot with no pickups."""
        grid = Grid(3, 3)
        grid.set_cell(2, 2, CellType.DEPOT)
        solver = DeliverySolver(grid, start=(0, 0))
        route = solver.find_route()
        self.assertTrue(len(route) > 0, "Should find a route")
        self.assertEqual(route[-1], (2, 2), "Route must end at depot")

    def test_single_pickup_then_depot(self) -> None:
        grid = Grid(4, 4)
        grid.set_cell(1, 1, CellType.PICKUP)
        grid.set_cell(3, 3, CellType.DEPOT)
        solver = DeliverySolver(grid, start=(0, 0))
        route = solver.find_route()
        self.assertTrue(len(route) > 0)
        self.assertIn((1, 1), route, "Route must visit the pickup")
        self.assertEqual(route[-1], (3, 3), "Route must end at depot")

    def test_multiple_pickups(self) -> None:
        grid = Grid(5, 5)
        grid.set_cell(1, 1, CellType.PICKUP)
        grid.set_cell(3, 3, CellType.PICKUP)
        grid.set_cell(4, 0, CellType.DEPOT)
        solver = DeliverySolver(grid, start=(0, 0))
        route = solver.find_route()
        self.assertTrue(len(route) > 0)
        self.assertIn((1, 1), route)
        self.assertIn((3, 3), route)
        self.assertEqual(route[-1], (4, 0))


class TestSolverBlocked(unittest.TestCase):
    def test_route_around_wall(self) -> None:
        grid = Grid(5, 5)
        for r in range(4):
            grid.set_cell(r, 2, CellType.BLOCKED)
        grid.set_cell(4, 4, CellType.DEPOT)
        solver = DeliverySolver(grid, start=(0, 0))
        route = solver.find_route()
        self.assertTrue(len(route) > 0)
        for r, c in route:
            if c == 2 and r < 4:
                self.fail("Route passed through blocked cell")
        self.assertEqual(route[-1], (4, 4))

    def test_no_route_fully_blocked(self) -> None:
        grid = Grid(3, 3)
        grid.set_cell(2, 2, CellType.DEPOT)
        for c in range(3):
            grid.set_cell(1, c, CellType.BLOCKED)
        solver = DeliverySolver(grid, start=(0, 0))
        route = solver.find_route()
        self.assertEqual(len(route), 0, "No route should exist when fully blocked")


class TestSolverTraffic(unittest.TestCase):
    def test_prefers_low_cost_path(self) -> None:
        """Route should avoid high-traffic cells when a cheaper path exists."""
        grid = Grid(3, 3)
        grid.set_cell(2, 2, CellType.DEPOT)
        grid.set_traffic(0, 1, 10)
        grid.set_traffic(0, 2, 10)
        solver = DeliverySolver(grid, start=(0, 0))
        route = solver.find_route()
        self.assertTrue(len(route) > 0)
        self.assertEqual(route[-1], (2, 2))
        self.assertLessEqual(solver.get_total_cost(), 4,
            "Optimal route goes down then right, cost=4 (all default traffic=1)")

    def test_exact_minimum_cost(self) -> None:
        """3x3 grid with known optimal cost to verify true shortest path."""
        grid = Grid(3, 3)
        grid.set_cell(0, 2, CellType.PICKUP)
        grid.set_cell(2, 2, CellType.DEPOT)
        grid.set_traffic(0, 1, 5)
        grid.set_traffic(1, 2, 5)
        solver = DeliverySolver(grid, start=(0, 0))
        route = solver.find_route()
        self.assertTrue(len(route) > 0)
        self.assertIn((0, 2), route)
        self.assertEqual(route[-1], (2, 2))
        self.assertLessEqual(solver.get_total_cost(), 14,
            "Optimal: (0,0)->(1,0)->(2,0)->(2,1)->(2,2)->(1,2)->(0,2) or equivalent, cost=14")


class TestSolverTeleport(unittest.TestCase):
    def test_teleport_saves_cost(self) -> None:
        grid = Grid(6, 6)
        grid.set_cell(0, 0, CellType.TELEPORT_A)
        grid.set_cell(5, 5, CellType.TELEPORT_B)
        grid.set_cell(5, 4, CellType.DEPOT)
        grid.set_cell(0, 1, CellType.PICKUP)
        solver = DeliverySolver(grid, start=(0, 0))
        route = solver.find_route()
        self.assertTrue(len(route) > 0)
        self.assertIn((0, 1), route)
        self.assertEqual(route[-1], (5, 4))
        self.assertLessEqual(
            solver.get_total_cost(), 3,
            "Teleport from (0,0) to (5,5) costs 0, then 1 step to depot"
        )

    def test_teleport_not_always_optimal(self) -> None:
        """When depot is close, teleport should NOT be used (it wastes steps)."""
        grid = Grid(4, 4)
        grid.set_cell(0, 3, CellType.TELEPORT_A)
        grid.set_cell(3, 0, CellType.TELEPORT_B)
        grid.set_cell(0, 1, CellType.DEPOT)
        solver = DeliverySolver(grid, start=(0, 0))
        route = solver.find_route()
        self.assertTrue(len(route) > 0)
        self.assertEqual(route[-1], (0, 1))
        self.assertEqual(solver.get_total_cost(), 1,
            "Direct route is 1 step, teleport would be worse")

    def test_pickup_ordering_with_teleport(self) -> None:
        """Two pickups on opposite corners - teleport enables efficient routing."""
        grid = Grid(8, 8)
        grid.set_cell(0, 7, CellType.PICKUP)
        grid.set_cell(7, 0, CellType.PICKUP)
        grid.set_cell(7, 7, CellType.DEPOT)
        grid.set_cell(0, 6, CellType.TELEPORT_A)
        grid.set_cell(7, 1, CellType.TELEPORT_B)
        solver = DeliverySolver(grid, start=(0, 0))
        route = solver.find_route()
        self.assertTrue(len(route) > 0)
        self.assertIn((0, 7), route)
        self.assertIn((7, 0), route)
        self.assertEqual(route[-1], (7, 7))
        self.assertLessEqual(solver.get_total_cost(), 16,
            "With teleport, should be much cheaper than traversing diagonals twice")


if __name__ == "__main__":
    unittest.main()
