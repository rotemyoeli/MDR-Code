import random
import functools
import memoization

import numpy
from typing import List, Tuple

from ..search import base_state


class State(base_state.State):
    @classmethod
    def goal(cls, board_width=3, board_height=3):
        board = numpy.arange(1, board_width * board_height+1, dtype=numpy.int8).reshape(board_width, board_height)
        board[board_width-1, board_height-1] = 0
        return cls(board)

    @classmethod
    def random(cls, board_width, board_height, steps = 100):
        s = cls.goal(board_width, board_height)
        for _ in range(steps):
            s.rand_move()

        return s

    @classmethod
    @memoization.cached
    def goal_internal_state(cls, board_width, board_height):
        s = cls.goal()
        return s._internal_state()

    def __init__(self, board: numpy.ndarray):
        self.board = board
        self.width = self.board.shape[0]
        self.height = self.board.shape[1]

    def clone(self):
        return self.__class__(self.board.copy())

    def _enum_move_targets(self):
        bx, by = self._get_blank_idx()

        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            new_bx = bx + dx
            new_by = by + dy

            if 0 <= new_bx < self.width and 0 <= new_by < self.height:
                yield new_bx, new_by

    def _get_blank_idx(self):
        blank_indices = numpy.where(self.board == 0)
        blank_idx = blank_indices[0][0], blank_indices[1][0]
        return blank_idx

    def _move(self, new_blank_idx):
        bx, by = self._get_blank_idx()
        new_bx, new_by = new_blank_idx

        self.board[bx, by] = self.board[new_bx, new_by]
        self.board[new_bx, new_by] = 0

    def rand_move(self):
        move = random.choice(list(self._enum_move_targets()))
        self._move(move)

    def expand(self) -> List[Tuple[int, "State"]]:
        for move_target in self._enum_move_targets():
            ns = self.clone()
            ns._move(move_target)
            yield 1, ns

    @property
    def is_goal(self):
        return self._internal_state() == self.goal_internal_state(*self.board.shape)

    def _internal_state(self):
        return self.board.tostring()

    def dump(self):
        separator_row = '-' * (6 * self.board.shape[1] + 3)
        print(separator_row)
        for i in range(self.board.shape[0]):
            row = '|  ' + ' | '.join(f'{self.board[i, j]:-3d}' for j in range(self.board.shape[1])) + '  |'
            print(row)
            print(separator_row)

def h_manhatten_distance(s: State):
    distance = 0
    for i, row in enumerate(s.board):
        for j, val in enumerate(row):
            if 0 == val:
                continue
            goal_val_i = (val - 1) // s.width
            goal_val_j = (val - 1) % s.width

            distance += abs(i-goal_val_i) + abs(j-goal_val_j)

    return distance