from typing import List

from . import agent
import copy


def _find_state(states, step):
    if not states:
        raise IndexError('states is empty')

    left = 0
    right = len(states) - 1

    if not states[left].step <= step <= states[right].step:
        raise IndexError(f'step {step} not between left.step, right.step')

    while left < right:
        ls = states[left]
        rs = states[right]

        if ls.step == step:
            return ls
        elif rs.step == step:
            return rs

        mid = (left + right) // 2
        ms = states[mid]
        if ms.step == step:
            return ms
        elif ms.step < step:
            left = mid
        else:
            right = mid

    raise IndexError(f'step {step} not found in states')


class AgentRepository:
    def __init__(self, agents: List[agent.Agent]):
        self.agents = agents

    def clone(self):
        return copy.deepcopy(self)

    def __iter__(self):
        return iter(self.agents)

    def get_occupied_cells_at_step(self, step_idx):
        occupied_cells = []
        for a in self:
            if not a.path:
                continue

            try:
                state = _find_state(a.path, step_idx)
            except IndexError:
                if a.goal_policy == agent.GoalPolicy.STAY_AT_GOAL:
                    last_state = a.get_last_state()
                    if not last_state:
                        continue
                    if last_state.is_goal:
                        occupied_cells.append(last_state.cell)

                continue

            occupied_cells.append(state.cell)

        return occupied_cells

    def get_agent_by_id(self, _id) -> agent.Agent:
        for a in self.agents:
            if a.id == _id:
                return a

        return None

    def get_adversarial_agents(self) -> List[agent.Agent]:
        return [a for a in self.agents if a.is_adversarial]

    def get_agent_cell_at_step(self, _agent: agent.Agent, step_idx: int):
        if _agent.path is None:
            return None

        try:
            state = _find_state(_agent.path, step_idx)
        except IndexError:
            return None

        return state.cell

    def get_makespan(self, only_non_adversarial=True):
        return max(len(a.path) for a in self.agents if not only_non_adversarial or not a.is_adversarial)

    def get_first_move_step(self, agent_id) -> int:

        agnt = self.get_agent_by_id(agent_id)

        for state in agnt.path:
            if state.cell != agnt.start_cell:
                return state.step

