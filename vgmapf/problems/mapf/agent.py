import copy

from enum import Enum
from dataclasses import dataclass, field
from typing import Tuple, List


class StartPolicy(Enum):
    STAY_AT_START = 1
    APPEAR_AT_START = 2


class GoalPolicy(Enum):
    STAY_AT_GOAL = 1
    DISAPPEAR_AT_GOAL = 2


class MotionEquation(Enum):
    MOVE_4_DIRECTIONS = 4
    MOVE_5_DIRECTIONS = 5
    MOVE_8_DIRECTIONS = 8
    MOVE_9_DIRECTIONS = 9


@dataclass
class Agent:
    id: int
    start_policy: StartPolicy
    goal_policy: GoalPolicy
    motion_equation: MotionEquation
    start_cell: Tuple[int, int]
    goal_cell: Tuple[int, int]

    must_reach_target: bool = True
    is_adversarial: bool = False
    damage_steps: int = 0
    step_size: int = 1
    initial_step: int = 0

    path: List["PathfindingState"] = None
    path_cost: float = 0.0
    expanded_nodes: int = 0
    constraints: List[Tuple[int, Tuple[int, int]]] = field(default_factory=lambda: [])


    @classmethod
    def from_dict(cls, d):
        a = cls(**d)
        return a

    def cells_path(self):
        return [x.cell for x in self.path]

    def get_last_state(self) -> "PathfindingState":
        if self.path is not None:
            return self.path[-1]

    def clear(self):
        self.path = None
        self.path_cost = 0.0
        self.expanded_nodes = 0

    def clone(self, clear_path=True):
        new_agent = self.__class__(**self.__dict__)
        if clear_path:
            new_agent.clear()

        return new_agent

    def to_dict(self):
        return copy.deepcopy(self.__dict__)