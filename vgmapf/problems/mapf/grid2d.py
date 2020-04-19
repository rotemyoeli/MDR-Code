import logging
import math
import pathlib
import random
from enum import Enum
from typing import Set, Tuple, List
import numpy
import io
from .agent import MotionEquation

SQRT2 = math.sqrt(2)

Cell = Tuple[int, int]


class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    UP_LEFT = 4
    UP_RIGHT = 5
    DOWN_LEFT = 6
    DOWN_RIGHT = 7
    SIZE = 8


class GridError(Exception):
    pass


class CellState:
    FREE = 0
    OCCUPIED = 1


LOG = logging.getLogger(__name__)


def get_direction_cells(first_cell: Cell, direction: Direction, num_of_cells: int) -> List:
    x, y = first_cell
    cells = []

    for i in range(1, num_of_cells + 1):
        if direction is Direction.UP:
            cells.append((x, y - i))
        elif direction is Direction.DOWN:
            cells.append((x, y + i))
        elif direction is Direction.LEFT:
            cells.append((x - i, y))
        elif direction is Direction.RIGHT:
            cells.append((x + i, y))
        elif direction is Direction.UP_RIGHT:
            cells.append((x + i, y - i))
        elif direction is Direction.UP_LEFT:
            cells.append((x - i, y - i))
        elif direction is Direction.DOWN_LEFT:
            cells.append((x - i, y + i))
        elif direction is Direction.DOWN_RIGHT:
            cells.append((x + i, y + i))
    return cells


class Grid2D:
    @classmethod
    def from_file(cls, map_path: pathlib.Path):
        # load the map file into numpy array
        with map_path.open('rt') as infile:
            return cls._from_file_like(infile)

    @classmethod
    def from_str(cls, map_str):
        f = io.StringIO(map_str)
        return cls._from_file_like(f)

    @classmethod
    def _from_file_like(cls, f):
        grid1 = numpy.array([list(line.strip()) for line in f.readlines()])
        grid1[grid1 == '@'] = CellState.OCCUPIED
        grid1[grid1 == 'T'] = CellState.OCCUPIED
        grid1[grid1 == '.'] = CellState.FREE
        grid = numpy.array(grid1.astype(numpy.int8))
        return cls(grid)

    def __init__(self, grid: numpy.ndarray):
        """
        a grid represents the map (as 2d-list of nodes).
        """
        self.height, self.width = grid.shape
        self._grid = grid

    def is_inside_grid(self, cell):
        """
        check, if field position is inside map
        :param x: x pos
        :param y: y pos
        :return:
        """
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.height

    def is_free(self, cell):
        x, y = cell
        return self._grid[y, x] == CellState.FREE

    def get_complete_radius(self, center_cell: Cell, radius_size: int):
        x_c, y_c = center_cell

        neighbors = list(((x, y) for x in range(x_c - radius_size, x_c + radius_size + 1)
                          for y in range(y_c - radius_size, y_c + radius_size + 1)))

        good_neighbors = [cell for cell in neighbors if
                          self.is_inside_grid(cell) and
                          self.is_free(cell)]

        return good_neighbors

    def get_cells_in_radius(self, center_cell: Cell, radius_size: int, motion_equation: MotionEquation) -> List:

        x, y = center_cell

        if motion_equation in (MotionEquation.MOVE_4_DIRECTIONS, MotionEquation.MOVE_5_DIRECTIONS):
            radius_cells = self.get_complete_radius(center_cell, radius_size - 1)
            radius_cells.append((x, y - radius_size))  # ↑
            radius_cells.append((x + radius_size, y))  # →
            radius_cells.append((x, y + radius_size))  # ↓
            radius_cells.append((x - radius_size, y))  # ←

        elif motion_equation in (MotionEquation.MOVE_8_DIRECTIONS, MotionEquation.MOVE_9_DIRECTIONS):
            radius_cells = self.get_complete_radius(center_cell, radius_size)

        else:
            raise Exception("Wrong MotionEquation supplied")

        return radius_cells

    def get_accessible_neighbors(self, cell: Cell, radius_cells=None,
                                 motion_equation: MotionEquation = MotionEquation.MOVE_5_DIRECTIONS,
                                 is_start_cell: bool = False):
        """
        get all neighbors of one node
        :param radius_cells:
        :param motion_equation:
        :param cell: a tuple with x y coordinate
        :param node: node
        """
        if radius_cells is None:
            radius_cells = []
        x, y = cell

        neighbors = []

        neighbors.append((x, y - 1))  # ↑
        neighbors.append((x + 1, y))  # →
        neighbors.append((x, y + 1))  # ↓
        neighbors.append((x - 1, y))  # ←

        if motion_equation == MotionEquation.MOVE_4_DIRECTIONS:
            pass

        if motion_equation in (MotionEquation.MOVE_5_DIRECTIONS, MotionEquation.MOVE_9_DIRECTIONS):
            neighbors.append((x, y))

        if motion_equation in (MotionEquation.MOVE_8_DIRECTIONS, MotionEquation.MOVE_9_DIRECTIONS):
            neighbors.append((x - 1, y - 1))  # ↖
            neighbors.append((x + 1, y - 1))  # ↗
            neighbors.append((x + 1, y + 1))  # ↘
            neighbors.append((x - 1, y + 1))  # ↙

        good_neighbors = [cell for cell in neighbors if
                          self.is_inside_grid(cell) and
                          self.is_free(cell) and
                          not self._has_conflict(cell, radius_cells)]

        # LOG.debug( f'Cell: {cell}\nNeighbors: {neighbors}\nGood neighbors: {good_neighbors}\nRadius cells: {radius_cells}')

        if not good_neighbors:
            if is_start_cell:
                # if no valid neighbors and the cell is the start cell of the agent - he can stay at this cell since
                # if he didn't make the first move - meaning he not in the game yet
                good_neighbors = [(x, y)]
            else:
                return None

        neighbors_with_distances = [(self.get_neighbor_distance((x, y), n), n) for n in good_neighbors]

        return neighbors_with_distances

    def to_str(self, path=None, start=None, end=None,
               border=False, start_chr='s', end_chr='g',
               path_chr='x', empty_chr='.', block_chr='@',
               show_weight=False):
        """
        create a printable string from the grid using ASCII characters

        :param path: list of nodes that show the path
        :param start: start node
        :param end: end node
        :param border: create a border around the grid
        :param start_chr: character for the start (default "s")
        :param end_chr: character for the destination (default "e")
        :param path_chr: character to show the path (default "x")
        :param empty_chr: character for empty fields (default " ")
        :param block_chr: character for blocking elements (default "#")
        :param show_weight: instead of empty_chr show the cost of each empty
                            field (shows a + if the value of weight is > 10)
        :return:
        """
        data = ''
        if border:
            data = '+{}+'.format('-' * self.width)
        for y in range(self.height):
            line = ''
            for x in range(self.width):
                cell = (x, y)
                if cell == start:
                    line += start_chr
                elif cell == end:
                    line += end_chr
                elif path and (x, y) in path:
                    line += path_chr
                elif self.is_free((x, y)):
                    line += empty_chr
                else:
                    line += block_chr  # blocked field
            if border:
                line = '|' + line + '|'
            if data:
                data += '\n'
            data += line
        if border:
            data += '\n+{}+'.format('-' * self.width)
        return data

    def _has_conflict(self, cell, other_cells):
        """
        Returns true if _cell_ is conflicting with any of the cells in _other_cells_
        """
        for o_cell in other_cells:
            if o_cell == cell:
                return True
        return False

    @classmethod
    def get_neighbor_distance(cls, cell_a, cell_b):
        """
        get the distance between cell_a, cell_b assuming they are neighbors (potentially diagonal)
        """
        if cell_a == cell_b:
            return 1

        xa, ya = cell_a
        xb, yb = cell_b

        if xb - xa == 0 or yb - ya == 0:
            # direct neighbor - distance is 1
            return 1
        else:
            return SQRT2

    def get_random_free_cell(self, excludes: Set[Cell] = None):
        if excludes is None:
            excludes = set()
        while True:
            cell = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
            if self.is_free(cell) and cell not in excludes:
                return cell

    def find_free_cells_in_row(self, first_cell: Cell, cells_to_get: int, extra_exclude_cells: Set[Cell] = None):
        if extra_exclude_cells is None:
            extra_exclude_cells = set()

        rand_direc = random.sample(range(0, Direction.SIZE.value), Direction.SIZE.value)

        for direc in rand_direc:
            neighbor_cells = get_direction_cells(first_cell, Direction(direc), cells_to_get)

            valid_cells = []

            for new_cell in neighbor_cells:
                if self.is_inside_grid(new_cell) and self.is_free(
                        new_cell) and new_cell not in extra_exclude_cells:
                    valid_cells.append(new_cell)

            if len(valid_cells) == cells_to_get:
                return valid_cells

        return None

    def find_free_cells_around(self, center_cell: Cell, cells_to_get: int, extra_exclude_cells: Set[Cell] = None,
                               radius: int = 1):
        if extra_exclude_cells is None:
            extra_exclude_cells = set()
        neighbor_cells = self.get_cells_in_radius(center_cell, radius, MotionEquation.MOVE_9_DIRECTIONS)
        try:
            neighbor_cells.remove(center_cell)
        except ValueError:
            pass
        random.shuffle(neighbor_cells)

        already_taken = set()

        found_cells = []

        while len(found_cells) < cells_to_get:
            try:
                new_cell = neighbor_cells.pop(0)
            except IndexError:
                raise GridError(f"Could not find {cells_to_get} free cells around {center_cell}")

            if self.is_inside_grid(new_cell) and self.is_free(
                    new_cell) and new_cell not in extra_exclude_cells and new_cell not in already_taken:
                found_cells.append(new_cell)

        return found_cells
