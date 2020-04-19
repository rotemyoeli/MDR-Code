import logging
import pathlib
from enum import Enum
from typing import List, Tuple, Iterator, Dict

from . import path_validation
from . import agent, paths_serializer
from . import agent_repository
from . import grid2d
from . import pathfinding

from ...search import base_algorithm

LOG = logging.getLogger(__name__)


class MapfState(pathfinding.State):
    def __init__(self, grid: grid2d.Grid2D, cell: Tuple[int, int], step: int, agent: agent.Agent,
                 agent_repo: agent_repository.AgentRepository,
                 adv_agent_radiuses: dict = None):
        super(MapfState, self).__init__(grid, cell, step, agent.start_cell, agent.goal_cell)

        self._agent = agent
        self._agent_repo = agent_repo
        self.step = step
        self.adv_agent_radiuses = adv_agent_radiuses or dict()

    def _internal_state(self):
        # return self.cell, self.step if self.is_start else self.cell
        return self.cell, self.step

    def clone_with_new_cell(self, cell, inc_step=True) -> "MapfState":
        return MapfState(self.grid, cell, self.step + 1 if inc_step else self.step, self._agent, self._agent_repo,
                         self.adv_agent_radiuses)

    def expand(self) -> List[Tuple[int, "MapfState"]]:

        dangerous_cells = self.robust_route_dangerous_cells()
        occupied_cells_by_other_paths = self._agent_repo.get_occupied_cells_at_step(self.step + 1)
        neighbor_cells = self.grid.get_accessible_neighbors(self.cell,
                                                            radius_cells=list(set(dangerous_cells +
                                                                                  occupied_cells_by_other_paths)),
                                                            motion_equation=self._agent.motion_equation,
                                                            is_start_cell=self.cell == self._agent.start_cell)
        if not neighbor_cells:
            return []

        return [(distance, self.clone_with_new_cell(n_cell, inc_step=True)) for distance, n_cell in neighbor_cells]

    def robust_route_dangerous_cells(self):
        dangerous_cells = []
        if self._agent.id not in self.adv_agent_radiuses:
            adversarial_agents = self._agent_repo.get_adversarial_agents()

            for adv_agnt in adversarial_agents:
                next_adv_cell: grid2d.Cell = self._agent_repo.get_agent_cell_at_step(adv_agnt, self.step + 1)
                if not next_adv_cell:
                    continue

                a_radius = self.adv_agent_radiuses.get(adv_agnt.id, 0)

                """ Use the following line for ONLINE mode where 1 step of distance is needed"""
                #a_radius = 1

                if not a_radius:
                    continue

                next_step = self.step + 1
                steps_left_to_finish = len(adv_agnt.path) - next_step

                if next_step >= a_radius and steps_left_to_finish > a_radius:
                    num_of_steps_to_stay_away = a_radius

                elif next_step < a_radius:
                    num_of_steps_to_stay_away = next_step

                else:
                    num_of_steps_to_stay_away = steps_left_to_finish

                dangerous_cells += self.grid.get_cells_in_radius(next_adv_cell, num_of_steps_to_stay_away,
                                                                 adv_agnt.motion_equation)

                # if the agent didn't move yet, the agent can safely stay in the start point and wait until
                # the adversarial agent will increase the gap
                if self.cell == self.start_cell and self.start_cell in dangerous_cells:
                    dangerous_cells.remove(self.start_cell)

        return dangerous_cells


class MapfFinder:
    def __init__(self, grid: grid2d.Grid2D, agents: List[agent.Agent],
                 adv_agent_radiuses: Dict[int, int] = None,
                 skip_path_finding_for_adv_agents=True):
        self.grid = grid
        self.agents_repo = agent_repository.AgentRepository(agents)
        self.adv_agent_radiuses = adv_agent_radiuses or dict()
        self.skip_path_finding_for_adv_agents = skip_path_finding_for_adv_agents

    def find_paths(self, searcher_class, searcher_kwargs) -> Dict[int, List[MapfState]]:
        """
        Finds paths for all agents and populate the _path_ field of all agent with the found path
        """
        for agnt in self.agents_repo:

            if agnt.id in self.adv_agent_radiuses and self.skip_path_finding_for_adv_agents:
                continue

            LOG.debug(f'START agent #{agnt.id}')
            nc = base_algorithm.NodeCounter()
            start_state = MapfState(self.grid, agnt.start_cell, step=agnt.initial_step, agent=agnt,
                                    agent_repo=self.agents_repo, adv_agent_radiuses=self.adv_agent_radiuses)

            if callable(searcher_kwargs):
                s_kwargs = searcher_kwargs(agnt)
            else:
                s_kwargs = searcher_kwargs

            searcher = searcher_class(
                start_state,
                observer=nc,
                **s_kwargs
            )
            assert isinstance(searcher, base_algorithm.SearchAlgorithm)

            if agnt.start_cell == agnt.goal_cell:
                path = None
                cost = 0
            else:
                path, cost = searcher.search()

            agnt.path = path
            agnt.path_cost = cost
            agnt.expanded_nodes = nc.count
            LOG.debug(f'FINISH agent #{agnt.id}')

        return {a.id: a.path for a in self.agents_repo}

    def get_paths(self):
        return [
            [(s.cell, s.step) for s in a.path]
            for a in self.agents_repo
        ]

    @property
    def agents(self) -> Iterator[agent.Agent]:
        return iter(self.agents_repo)

    def save_paths(self, target_path: pathlib.Path, metadata: dict):
        paths_serializer.dump(target_path, list(self.agents), grid=self.grid, metadata = metadata)

    def validate_paths(self, raise_error=True) -> None:
        path_validation.validate_paths(self.grid, self.agents_repo, raise_error)
