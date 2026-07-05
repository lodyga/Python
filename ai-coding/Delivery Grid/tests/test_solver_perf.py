"""Performance tests for DeliverySolver  -  Phase 3 optimization gate."""

import unittest
import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from grid import Grid, CellType
from solver import DeliverySolver


class TestSolverPerformance(unittest.TestCase):
    def test_large_grid_performance(self) -> None:
        """50x50 grid with 5 pickups  -  must complete under 5s."""
        grid = Grid(50, 50)
        grid.set_cell(10, 10, CellType.PICKUP)
        grid.set_cell(20, 30, CellType.PICKUP)
        grid.set_cell(30, 10, CellType.PICKUP)
        grid.set_cell(40, 40, CellType.PICKUP)
        grid.set_cell(5, 45, CellType.PICKUP)
        grid.set_cell(49, 49, CellType.DEPOT)

        solver = DeliverySolver(grid, start=(0, 0))
        start_time = time.time()
        route = solver.find_route()
        elapsed = time.time() - start_time

        self.assertTrue(len(route) > 0, "Must find a route")
        self.assertEqual(route[-1], (49, 49))
        self.assertLess(elapsed, 5.0, f"Took {elapsed:.2f}s, must be under 5s")

    def test_large_grid_with_traffic_and_teleport(self) -> None:
        """50x50 grid with high-traffic zones and teleport  -  tests optimization quality."""
        grid = Grid(50, 50)
        for r in range(50):
            for c in range(25, 30):
                grid.set_traffic(r, c, 8)

        grid.set_cell(10, 10, CellType.PICKUP)
        grid.set_cell(10, 40, CellType.PICKUP)
        grid.set_cell(40, 10, CellType.PICKUP)
        grid.set_cell(40, 40, CellType.PICKUP)
        grid.set_cell(49, 49, CellType.DEPOT)
        grid.set_cell(0, 24, CellType.TELEPORT_A)
        grid.set_cell(0, 30, CellType.TELEPORT_B)

        solver = DeliverySolver(grid, start=(0, 0))
        start_time = time.time()
        route = solver.find_route()
        elapsed = time.time() - start_time

        self.assertTrue(len(route) > 0, "Must find a route")
        self.assertEqual(route[-1], (49, 49))
        self.assertLess(elapsed, 5.0, f"Took {elapsed:.2f}s, must be under 5s")


if __name__ == "__main__":
    unittest.main()
