import logging
import math
from enum import Enum
from typing import List, Tuple, Dict
from abc import ABC, abstractmethod
from dataclasses import dataclass

from . import multi_agent_pathfinding
from . import heuristics
from . import agent
from . import grid2d
from .grid2d import Cell
from ...search import base_state, base_algorithm
from ...search.algorithms import algo_utils
from . import dijkstra
from ...search.base_algorithm import NotFoundError

PathsType = Dict[int, List[multi_agent_pathfinding.MapfState]]
AgentsType = List[agent.Agent]

LOG = logging.getLogger(__name__)

STATE_LOG_INTERVAL = 100


class RobustPathMode(Enum):
    DISABLE = 0
    OFFLINE = 1
    ONLINE = 2
    ONLINE_CONST = 3


class _DatabaseHeuristic(object):
    def __init__(self, goal_to_db):
        self._goal_to_db = goal_to_db

    def __call__(self, s: multi_agent_pathfinding.MapfState):
        db = self._goal_to_db[s.goal_cell]
        h = db[s.cell]
        return h


@dataclass
class _PlanMetadata:
    expanded_states: int
    total_goals: int
    visited_states: int


class AAPState(base_state.State):
    def __init__(self, grid: grid2d.Grid2D, agents: AgentsType, adv_agent: agent.Agent,
                 paths: PathsType, step: int, ds: int,
                 searcher_class, searcher_kwargs,
                 robust_mode: RobustPathMode,
                 robust_radius: int = None,
                 first_allowed_damage_step: int = 0):
        self.grid = grid
        self.agents = agents
        self.adv_agent = adv_agent

        self.paths = paths
        self.step = step
        self.ds = ds  # remaining damage steps

        self.searcher_class = searcher_class
        self.searcher_kwargs = searcher_kwargs
        self.robust_mode = robust_mode
        self.first_allowed_damage_step = first_allowed_damage_step

        default_robust_radiuses = {
            RobustPathMode.ONLINE: lambda _ds: _ds * 2,
            RobustPathMode.ONLINE_CONST: 2
        }

        if robust_radius is None:
            robust_radius = default_robust_radiuses.get(self.robust_mode, None)

        self.robust_radius = robust_radius

    def __str__(self):
        return f'MdrState< step={self.step}, cell={self.cell}, ds={self.ds}'

    __repr__ = __str__

    @property
    def cell(self) -> Cell:
        try:
            return self.paths[self.adv_agent.id][self.step].cell
        except IndexError:
            return self.paths[self.adv_agent.id][-1].cell

    def cells_at_step(self, step) -> Dict[int, Cell]:
        cells = {}
        for aid, p in self.paths.items():
            try:
                cells[aid] = p[step].cell
            except IndexError:
                cells[aid] = None

        return cells

    def clone(self):
        return self.__class__(self.grid, self.agents, self.adv_agent, self.paths, self.step, self.ds,
                              self.searcher_class, self.searcher_kwargs, self.robust_mode, self.robust_radius,
                              self.first_allowed_damage_step)

    def expand(self) -> List[Tuple[int, "AAPState"]]:
        try:
            cur_cell = self.paths[self.adv_agent.id][self.step].cell
        except IndexError:
            cur_cell = self.paths[self.adv_agent.id][-1].cell

        if cur_cell == self.adv_agent.goal_cell:
            return

        if self.ds == 0 or self.step < self.first_allowed_damage_step:
            new_state = self.clone()
            new_state.step = self.step + 1

            yield None, new_state
            return

        try:
            next_planned_cell = self.paths[self.adv_agent.id][self.step + 1].cell
        except IndexError:
            next_planned_cell = self.paths[self.adv_agent.id][-1].cell

        planned_step_angle = math.degrees(
            math.atan2(next_planned_cell[1] - cur_cell[1], next_planned_cell[0] - cur_cell[0]))
        distances_and_neighbors = self.grid.get_accessible_neighbors(cur_cell,
                                                                     motion_equation=self.adv_agent.motion_equation)

        for d, neighbor_cell in distances_and_neighbors:
            neighbor_step_angle = math.degrees(
                math.atan2(neighbor_cell[1] - cur_cell[1], neighbor_cell[0] - cur_cell[0]))

            angle_diff = (neighbor_step_angle - planned_step_angle + 180 + 360) % 360 - 180

            if angle_diff < -90 or angle_diff > 90:
                continue

            new_state = self._create_new_state_for_adv_move(neighbor_cell)
            yield None, new_state

    def _create_new_state_for_adv_move(self, next_cell):
        new_state = self.clone()
        new_state.step = self.step + 1

        try:
            org_next_cell = self.paths[self.adv_agent.id][new_state.step].cell
        except IndexError:
            org_next_cell = None

        if org_next_cell != next_cell:
            new_state.ds -= 1
            assert new_state.ds >= 0
            new_state.paths = self._compute_new_paths(next_cell, org_next_cell, new_state.ds)

        return new_state

    def _compute_new_paths_robust_offline(self, next_cell, org_next_cell, new_state_ds: int) -> PathsType:
        a = self.adv_agent.clone()

        a.start_cell = next_cell
        a.initial_step = self.step + 1
        a.goal_cell = org_next_cell

        mf = multi_agent_pathfinding.MapfFinder(self.grid, [a])
        partial_new_paths = mf.find_paths(self.searcher_class,
                                          lambda agnt: dict(h_func=heuristics.get_good_manhatten_like_heuristic(agnt)))

        # mf.validate_paths(raise_error=False)

        path_from_next_to_org_next = partial_new_paths[a.id]
        last_state = path_from_next_to_org_next[-1]

        new_paths = _merge_paths(self.paths, partial_new_paths)

        a.clear()
        a.start_cell = last_state.cell
        a.initial_step = last_state.step
        a.goal_cell = self.adv_agent.goal_cell

        mf = multi_agent_pathfinding.MapfFinder(self.grid, [a])
        partial_new_paths = mf.find_paths(self.searcher_class, self.searcher_kwargs)

        new_paths = _merge_paths(new_paths, partial_new_paths)

        return new_paths

    def _compute_new_paths(self, next_cell, org_next_cell, new_state_ds: int) -> PathsType:
        """Compute and return new paths for all agents starting step self.step and for current adversarial agent
        starting with self.step+1 """
        if self.robust_mode == RobustPathMode.OFFLINE:
            return self._compute_new_paths_robust_offline(next_cell, org_next_cell, new_state_ds)

        agents = [a.clone() for a in self.agents]

        for a in agents:
            if a.id == self.adv_agent.id:
                a.start_cell = next_cell
                a.initial_step = self.step + 1
            elif self.step < len(self.paths[a.id]):
                a.start_cell = self.paths[a.id][self.step].cell
                a.initial_step = self.step
            else:  # self.step >= len(self.paths[a.id])
                a.start_cell = a.goal_cell
                a.initial_step = self.step

        adv_agent_radiuses = dict()
        if self.robust_mode == RobustPathMode.ONLINE:
            adv_agent_radiuses[self.adv_agent.id] = new_state_ds * 2

            """For distance of 1 step in ONLINE mode use the following line"""
            #adv_agent_radiuses[self.adv_agent.id] = 1


        elif self.robust_mode == RobustPathMode.ONLINE_CONST:
            adv_agent_radiuses[self.adv_agent.id] = self.robust_radius if new_state_ds > 0 else 0

        mf = multi_agent_pathfinding.MapfFinder(self.grid, agents, adv_agent_radiuses,
                                                skip_path_finding_for_adv_agents=False)
        partial_new_paths = mf.find_paths(self.searcher_class, self.searcher_kwargs)

        # mf.validate_paths()

        new_paths = _merge_paths(self.paths, partial_new_paths)
        return new_paths

    def _internal_state(self):
        paths = [(aid, tuple(p)) for aid, p in self.paths.items()]
        paths.sort(key=lambda x: x[0])
        paths = tuple(paths)
        return paths, self.step, self.ds, self.adv_agent.id

    @abstractmethod
    def g(self):
        pass

    @abstractmethod
    def f(self):
        pass

    @property
    @abstractmethod
    def is_goal(self):
        pass


class AdversarialAgentPlanner:
    def __init__(self,
                 grid: grid2d.Grid2D,
                 agents: AgentsType,
                 searcher_class,
                 searcher_kwargs,
                 state_factory,
                 put_adv_agents_in_front=True,
                 robust_mode: RobustPathMode = RobustPathMode.DISABLE,
                 robust_radius=None,
                 stop_at_first_goal = False,
                 prune_state_above_current_min_goal = True):
        self.grid = grid
        self.agents = agents
        self.searcher_class = searcher_class
        self.searcher_kwargs = searcher_kwargs
        self.state_factory = state_factory
        self.robust_mode = robust_mode
        self.robust_radius = robust_radius
        self.stop_at_first_goal = stop_at_first_goal
        self.prune_state_above_current_min_goal = prune_state_above_current_min_goal

        if put_adv_agents_in_front:
            adv_agents = [a for a in agents if a.is_adversarial]
            non_adv_agents = [a for a in agents if not a.is_adversarial]
            self.agents = adv_agents + non_adv_agents

    def find(self) -> Tuple[AAPState, _PlanMetadata]:
        # Find initial paths and populate agent.path for each agent
        # mf = multi_agent_pathfinding.MapfFinder(self.grid, self.agents)
        # mf.find_paths(self.searcher_class, self.searcher_kwargs)

        adv_agent = [a for a in self.agents if a.is_adversarial][0]
        mdr_goal_state, info = self._find_mdr_for_adversarial_agent(adv_agent)
        return mdr_goal_state, info

    def _find_mdr_for_adversarial_agent(self, agnt: agent.Agent) -> Tuple[base_state.State, _PlanMetadata]:
        goal_to_db = dict()
        for a in self.agents:
            LOG.debug(f'STARTED computing dijkstra db heuristic for agent: {a}')
            db = dijkstra.dijkstra(self.grid, a.goal_cell, a.motion_equation)
            goal_to_db[a.goal_cell] = db
            LOG.debug(f'FINISHED computing dijkstra db heuristic for agent: {a}')
        db_heuristic = _DatabaseHeuristic(goal_to_db)

        start_state = self.state_factory(
            self.grid,
            self.agents,
            agnt,
            {a.id: a.path for a in self.agents},
            0,
            agnt.damage_steps,
            self.searcher_class,
            dict(h_func=db_heuristic),
            self.robust_mode,
            self.robust_radius
        )

        prepo = algo_utils.NodeRepository()
        open_set = algo_utils.HashedPriorityQueue(prepo.get_node_f)

        prepo.add_root(start_state)
        prepo.set_node_f(start_state, start_state.f())
        open_set.put(start_state)

        min_goal_value = 0  # = start_state.g()

        expanded_states = 0
        visited_states = 0
        goals = []

        while not open_set.empty():
            cur_node, f = open_set.get()  # type: (base_state.State, int)

            expanded_states += 1
            visited_states += 1
            if 0 == expanded_states % STATE_LOG_INTERVAL:
                LOG.debug(f'MDR states expanded: {expanded_states}, min f in OPEN: {f}, cur_node: {cur_node}')

            # LOG.debug(f'cur_node:: step={cur_node.step} | cell = {cur_node.cell} | ds = {cur_node.ds} | f = {f:.2f} | min_goal_value = {min_goal_value:.2f}')

            if self.prune_state_above_current_min_goal and f > min_goal_value:  # This means that this state will not lead to an outcome better than the current best outcome
                continue

            if cur_node.is_goal:
                goals.append(cur_node)
                if self.stop_at_first_goal:
                    break
                LOG.debug(
                    f'New goal state found: {cur_node}, total: {len(goals)} goals found, makespan: {cur_node.g()}, original: {start_state.g()}')
                continue

            for d, new_node in cur_node.expand():
                if 0 == expanded_states % STATE_LOG_INTERVAL:
                    LOG.debug(f'new_node: {new_node}')

                visited_states += 1

                new_f = new_node.f()
                prepo.add(new_node, cur_node)
                prepo.set_node_f(new_node, new_f)

                if self.prune_state_above_current_min_goal and new_f > min_goal_value:  # This means that this state will not lead to an outcome better than the current best outcome
                    continue

                if new_node.is_goal:
                    goals.append(new_node)
                    if self.stop_at_first_goal:
                        break
                    LOG.debug(
                        f'New goal state found: {new_node}, total: {len(goals)} goals found, makespan: {new_node.g()}, original: {start_state.g()}')

                    new_node_cost = new_node.g()
                    if new_node_cost < min_goal_value:
                        min_goal_value = new_node_cost
                else:
                    open_set.put(new_node)

            if goals and self.stop_at_first_goal:
                break

        if not goals:
            raise NotFoundError()

        best_goal = min(goals, key=lambda goal: goal.g())

        return best_goal, _PlanMetadata(expanded_states, len(goals), visited_states)


def get_makespan(paths: PathsType, agents: AgentsType, only_non_adversarial=True):
    agents_mapping = {a.id: a for a in agents}
    return max(len(p) for aid, p in paths.items() if not only_non_adversarial or not agents_mapping[aid].is_adversarial)
    # return max(len(p) for aid, p in paths.items()) #if not only_non_adversarial or not agents_mapping[aid].is_adversarial)


def _get_cells_at_step(paths: PathsType, step: int):
    cells = dict()

    for aid, p in paths.items():
        if step < len(p):
            cell = p[step]
        else:
            cell = p[-1]

        cells[aid] = cell

    return cells


def _merge_paths(paths1: PathsType, paths2: PathsType):
    paths = dict()

    for aid, p1 in paths1.items():
        p2 = paths2.get(aid)
        if not p2:
            p = p1[:]
        else:
            p = p1[:p2[0].step]
            # TODO:, this assert fails - maybe because of using the faster _internal_state implementation that returns just cell without step?
            try:
                assert not p or (p[-1].step + 1 == p2[0].step)
            except AssertionError:
                from IPython import embed;
                embed()
            p += p2

        paths[aid] = p

    return paths
