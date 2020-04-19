import logging
import pathlib
from typing import List, Tuple
import copy
import pandas
from . import agent
from . import agent_repository
from . import grid2d
from . import pathfinding
from .path_validation import validate_paths
from ...search import base_algorithm
from ...search.algorithms import algo_utils

import hashlib

LOG = logging.getLogger(__name__)


class CbsHashedPriorityQueue(algo_utils.HashedPriorityQueue):
    def get_with_break_even(self, compare_func, break_even_func):
        node = min(self.nodes, key=lambda n: compare_func(n))
        min_val = compare_func(node)
        list_of_min_elements = [_node for _node in self.nodes if _node.total_cost == min_val]

        node = min(list_of_min_elements, key=lambda n: break_even_func(n))

        self.remove(node)

        return node, min_val

    def __contains__(self, node):
        return [n for n in self.nodes if n.id == node.id]


class CbsState(pathfinding.State):
    def __init__(self, grid: grid2d.Grid2D, cell: Tuple[int, int], step: int, agent: agent.Agent):
        super(CbsState, self).__init__(grid, cell, step, agent.start_cell, agent.goal_cell)

        self.agent = agent

    def __str__(self):
        return f'(({self.x, self.y}), {self.step})'

    __repr__ = __str__

    def _internal_state(self):
        return self.cell, self.step

    def clone_with_new_cell(self, cell, inc_step=True) -> "CbsState":
        return CbsState(self.grid, cell, self.step + 1 if inc_step else self.step, self.agent)

    def expand(self) -> List[Tuple[int, "CbsState"]]:
        valid_constraint_for_next_step = []
        for cell, step in self.agent.constraints:
            if step == self.step + 1:
                valid_constraint_for_next_step.append(cell)

        neighbor_cells = self.grid.get_accessible_neighbors(self.cell, valid_constraint_for_next_step,
                                                            self.agent.motion_equation,
                                                            is_start_cell=self.cell == self.agent.start_cell)

        return [(distance, self.clone_with_new_cell(n_cell, inc_step=True)) for distance, n_cell in neighbor_cells]


class CtNode:
    def __init__(self, agent_repo: agent_repository.AgentRepository):
        self._agent_repo = agent_repo
        self.total_cost = 0
        self.total_expend_count = 0
        self.total_constraint_size = 0
        self.id = self.ct_hash()
        self.update_total_params()

    def ct_hash(self):
        m = hashlib.sha1()

        for ag in self._agent_repo.agents:
            m.update(str(ag.path).encode('UTF-8'))
            m.update(str(ag.constraints).encode('UTF-8'))
            m.update(str(ag.path_cost).encode('UTF-8'))
            m.update(str(ag.expanded_nodes).encode('UTF-8'))

        return m.hexdigest()

    def get_total_cost(self):
        return self.total_cost

    def get_total_constraint_size(self):
        return self.total_constraint_size

    def get_total_expend_count(self):
        return self.total_expend_count

    def get_conflict(self):
        for agent_a in self.agent_repo.agents:
            for agent_b in self.agent_repo.agents:
                if agent_a == agent_b:
                    continue
                for i in range(min(len(agent_a.path), len(agent_b.path))):
                    if agent_a.path[i].cell == agent_b.path[i].cell and agent_a.path[i].step == agent_b.path[i].step:
                        return agent_a, agent_b, agent_a.path[i].cell, agent_a.path[i].step

        return None

    def update_total_params(self):
        self.total_cost = 0
        self.total_expend_count = 0
        self.total_constraint_size = 0
        for a in self._agent_repo:
            self.total_cost += a.path_cost
            self.total_expend_count += a.expanded_nodes
            self.total_constraint_size += len(a.constraints)

    def clone(self):
        return copy.deepcopy(self)

    @property
    def agent_repo(self):
        return self._agent_repo


class CbsMafpFinder:
    def __init__(self, grid: grid2d.Grid2D):
        self.grid = grid

    def __search_path_individually(self, _agent: agent.Agent, searcher_class, searcher_kwargs) -> Tuple[
        List, int, int]:

        start_state = CbsState(self.grid, _agent.start_cell, 0, _agent)

        nc = base_algorithm.NodeCounter()

        if callable(searcher_kwargs):
            s_kwargs = searcher_kwargs(_agent)
        else:
            s_kwargs = searcher_kwargs

        searcher = searcher_class(
            start_state,
            observer=nc,
            **s_kwargs
        )
        assert isinstance(searcher, base_algorithm.SearchAlgorithm)

        path, cost = searcher.search()
        return path, cost, nc.count

    def expend_node(self, _agent, node_to_expend, new_constraints, searcher_class, searcher_kwargs):
        new_node = node_to_expend.clone()  # type: CtNode
        a = new_node.agent_repo.get_agent_by_id(_agent.id)
        a.constraints.append(new_constraints)
        a.path, a.path_cost, a.expanded_nodes = self.__search_path_individually(a,
                                                                                searcher_class,
                                                                                searcher_kwargs)
        new_node.update_total_params()

        return new_node

    def find_path(self, agents_repo: agent_repository.AgentRepository, searcher_class, searcher_kwargs) -> \
            (agent_repository.AgentRepository, int):

        open_set = CbsHashedPriorityQueue()
        for agnt in agents_repo.agents:
            agnt.path, agnt.path_cost, agnt.expanded_nodes = self.__search_path_individually(agnt,
                                                                                             searcher_class,
                                                                                             searcher_kwargs)
        root = CtNode(agents_repo.clone())

        open_set.put(root)

        while not open_set.empty():
            node, cost = open_set.get_with_break_even(compare_func=lambda n: n.get_total_cost(),
                                                      break_even_func=lambda n: n.get_total_constraint_size())

            try:
                agent_a, agent_b, cell, step = node.get_conflict()
                new_constraints = (cell, step)
            except Exception:
                return node.agent_repo, cost

            new_node = self.expend_node(agent_a, node, new_constraints, searcher_class, searcher_kwargs)

            if new_node not in open_set:
                open_set.put(new_node)

            new_node = self.expend_node(agent_b, node, new_constraints, searcher_class, searcher_kwargs)

            if new_node not in open_set:
                open_set.put(new_node)

    @staticmethod
    def save_paths(agent_repo: agent_repository.AgentRepository, target_path: pathlib.Path):
        paths_for_csv = [
            [((s.x, s.y), s.step) for s in a.path]
            for a in agent_repo.agents
        ]

        df = pandas.DataFrame(paths_for_csv)
        df.to_csv(target_path, index=False, header=False)

    @staticmethod
    def validate_paths(grid: grid2d.Grid2D, agent_repo: agent_repository.AgentRepository, raise_error=True):
        validate_paths(grid, agent_repo, raise_error)
