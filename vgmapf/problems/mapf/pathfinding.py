from typing import List, Tuple

from ...search import base_state
from . import grid2d


class PathfindingState(base_state.State):
    def __init__(self, grid: grid2d.Grid2D, cell: Tuple[int, int], step: int, start_cell: Tuple[int, int],
                 goal_cell: Tuple[int, int], extra_occupied_cells: List[Tuple[int, int]] = None):
        self.grid = grid
        self.cell = cell
        self.step = step
        self.start_cell = start_cell
        self.goal_cell = goal_cell
        self.extra_occupied_cells = extra_occupied_cells

    def __str__(self):
        return f'(({self.x, self.y}), {self.step})'

    def __repr__(self):
        return f'{self.__class__.__name__}{self}'

    def clone_with_new_cell(self, cell) -> "PathfindingState":
        return PathfindingState(self.grid, cell, self.step+1, self.start_cell, self.goal_cell, self.extra_occupied_cells)

    def expand(self) -> List[Tuple[int, "State"]]:
        neighbor_cells = self.grid.get_accessible_neighbors(self.cell, self.extra_occupied_cells)
        return [(distance, self.clone_with_new_cell(n_cell)) for distance, n_cell in neighbor_cells]

    def clone(self):
        return self.clone_with_new_cell(self.cell)

    @property
    def is_goal(self):
        return self.cell == self.goal_cell

    @property
    def is_start(self):
        return self.cell == self.start_cell

    def _internal_state(self):
        return self.cell

    @property
    def x(self):
        return self.cell[0]

    @property
    def y(self):
        return self.cell[1]

    @property
    def gx(self):
        return self.goal_cell[0]

    @property
    def gy(self):
        return self.goal_cell[1]


State = PathfindingState
