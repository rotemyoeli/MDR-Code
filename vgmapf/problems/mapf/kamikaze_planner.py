import sys
from dataclasses import dataclass

from . import adversarial_agent_planner, grid2d
from .adversarial_agent_planner import AgentsType, RobustPathMode, NotFoundError
from . import heuristics
from .grid2d import Cell


class CollisionNotFoundError(Exception):
    pass


@dataclass
class Collision:
    target_agent_id: int
    step: int
    cell: Cell

class KamikazeState(adversarial_agent_planner.AAPState):
    def g(self):
        return self.adv_agent.damage_steps - self.ds  # How many damage steps have we taken already?

    def h(self):
        min_distance = sys.maxsize
        for step in range(self.first_allowed_damage_step, len(self.paths[self.adv_agent.id])):
            cells = self.cells_at_step(step)

            adv_cell = cells[self.adv_agent.id]

            for aid, cell in cells.items():
                if aid == self.adv_agent.id:
                    continue
                if cell is None:
                    continue

                distance = heuristics.octile_distance(abs(adv_cell[0] - cell[0]), abs(adv_cell[1] - cell[1]))
                if distance < min_distance:
                    min_distance = distance
                    if min_distance == 0:
                        break
            if 0 == min_distance:
                break

        return min_distance

    def f(self):
        return self.g() + self.h()

    @property
    def is_goal(self):
        return self.h() == 0

    def get_collision(self):
        for step in range(self.first_allowed_damage_step, len(self.paths[self.adv_agent.id])):
            cells = self.cells_at_step(step)

            adv_cell = cells[self.adv_agent.id]

            for aid, cell in cells.items():
                if aid == self.adv_agent.id:
                    continue
                if cell is None:
                    continue

                distance = heuristics.octile_distance(abs(adv_cell[0] - cell[0]), abs(adv_cell[1] - cell[1]))
                if 0 == distance:
                    return Collision(aid, step, cell)

        raise CollisionNotFoundError()


class KamikazePlanner(adversarial_agent_planner.AdversarialAgentPlanner):
    def __init__(self,
                 grid: grid2d.Grid2D,
                 agents: AgentsType,
                 searcher_class,
                 searcher_kwargs,
                 robust_mode: RobustPathMode = RobustPathMode.DISABLE,
                 put_adv_agents_in_front=True,
                 robust_radius=None,
                 first_allowed_damage_step = 5):
        super(KamikazePlanner, self).__init__(
            grid,
            agents,
            searcher_class,
            searcher_kwargs,
            lambda *args, **kwargs: KamikazeState(*args, **kwargs, first_allowed_damage_step = first_allowed_damage_step),
            put_adv_agents_in_front,
            robust_mode,
            robust_radius,
            stop_at_first_goal=True,
            prune_state_above_current_min_goal=False
        )
