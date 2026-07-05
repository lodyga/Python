"""Run this to see how your code is called."""

from grid import Grid, CellType
from solver import DeliverySolver


def create_sample_grid() -> Grid:
    grid = Grid(6, 6)
    grid.set_cell(0, 5, CellType.DEPOT)
    grid.set_cell(2, 2, CellType.PICKUP)
    grid.set_cell(4, 4, CellType.PICKUP)
    grid.set_cell(1, 3, CellType.BLOCKED)
    grid.set_cell(2, 3, CellType.BLOCKED)
    grid.set_cell(3, 3, CellType.BLOCKED)
    grid.set_cell(1, 0, CellType.TELEPORT_A)
    grid.set_cell(5, 5, CellType.TELEPORT_B)
    grid.set_traffic(4, 2, 3)
    grid.set_traffic(4, 3, 3)
    return grid


def main() -> None:
    grid = create_sample_grid()
    solver = DeliverySolver(grid, start=(5, 0))
    route = solver.find_route()

    if route:
        print(f"Route found with cost {solver.get_total_cost()}:")
        for r, c in route:
            cell = grid.get_cell(r, c)
            label = cell.name if cell != CellType.EMPTY else "."
            print(f"  ({r},{c}) {label}")
    else:
        print("No valid route found.")


if __name__ == "__main__":
    main()
