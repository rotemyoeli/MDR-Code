import logging
from collections import defaultdict
from typing import List, Tuple
from vgmapf.problems.mapf import grid2d, agent_repository
from vgmapf.problems.mapf.agent import MotionEquation, GoalPolicy
from . import agent

LOG = logging.getLogger(__name__)


def validate_agents_and_map_collusion(grid: grid2d.Grid2D, agent_repo: agent_repository.AgentRepository,
                   raise_error=True):
    dict_steps = defaultdict(list)
    is_start_policy_appear_at_start = False

    for agnt in agent_repo.agents:
        if not is_start_policy_appear_at_start and agnt.start_policy == agent.StartPolicy.APPEAR_AT_START:
            is_start_policy_appear_at_start = True

        if is_start_policy_appear_at_start and agnt.start_policy != agent.StartPolicy.APPEAR_AT_START:
            str_error = 'Not all agents start policy is set to StartPolicy.APPEAR_AT_START!'
            if raise_error:
                raise Exception(str_error)
            else:
                LOG.error(str_error)

        for state in agnt.path:
            dict_steps[state.step].append((agnt.id, state.cell))

    for step, cords_list in dict_steps.items():
        agnt_ids, cords = zip(*cords_list)
        cords: List[Tuple[int, int]]
        seen_cord = list()
        seen_agnt = list()

        # for all agents that are finished and the policy is stay at goal - append to seen
        for a in agent_repo.agents:
            if step >= len(a.path) and a.goal_policy == GoalPolicy.STAY_AT_GOAL:
                seen_cord.append(a.goal_cell)
                seen_agnt.append(a.id)

        if step == 0:
            if is_start_policy_appear_at_start:
                if len(set(cords)) == 1:
                    continue
                else:
                    str_error = 'Not all agents with start policy StartPolicy.APPEAR_AT_START has the same starting ' \
                                'point:{}'.format(cords)
                    if raise_error:
                        raise Exception(str_error)
                    else:
                        LOG.error(str_error)

        for i, cord in enumerate(cords):
            if cord not in seen_cord:
                seen_cord.append(cord)
                seen_agnt.append(agnt_ids[i])
            else:
                if agent_repo.get_first_move_step(agnt_ids[i]) > step or \
                        agent_repo.get_first_move_step(seen_agnt[seen_cord.index(cord)]) > step:
                    LOG.debug(f"Overlaping agent {agnt_ids[i]} and {seen_agnt[seen_cord.index(cord)]} "
                              f"at step {step}")
                else:
                    str_error = 'Collision detected at agent {0} and {1} at cell {2} at step {3}'\
                        .format(str(agnt_ids[i]), str(seen_agnt[seen_cord.index(cord)]), str(cord), str(step))
                    if raise_error:
                        raise Exception(str_error)
                    else:
                        LOG.error(str_error)

            if not grid.is_free(cord):
                str_error = "Agent: {0} collides at a wall at coordinate: {1} at step {2}".format(str(agnt_ids[i]),
                                                                                      str(cord), str(step))
                if raise_error:
                    raise Exception(str_error)
                else:
                    LOG.error(str_error)


def validate_next_step_valid(agent_id: int, curr_cord: Tuple[int, int], next_cord: Tuple[int, int], motion_eq: MotionEquation, raise_error):
    x_curr, y_curr = curr_cord
    x_next, y_next = next_cord

    str_error = "Agent ID: {} violated the motion equation: {} from step {} to step {}"

    if motion_eq == MotionEquation.MOVE_4_DIRECTIONS or motion_eq == MotionEquation.MOVE_8_DIRECTIONS:
        if x_curr == x_next and y_curr == y_next:
            if raise_error:
                raise str_error.format(agent_id, motion_eq, curr_cord, next_cord)
            else:
                LOG.error(str_error.format(agent_id, motion_eq, curr_cord, next_cord))

    if motion_eq == MotionEquation.MOVE_4_DIRECTIONS or motion_eq == MotionEquation.MOVE_5_DIRECTIONS:
        if abs(x_curr - x_next) > 0 and abs(y_curr - y_next) > 0:
            if raise_error:
                raise str_error.format(agent_id, motion_eq, curr_cord, next_cord)
            else:
                LOG.error(str_error.format(agent_id, motion_eq, curr_cord, next_cord))


def validate_motion_equation(agent_repo: agent_repository.AgentRepository, raise_error):

    for agent in agent_repo:
        for i, state in enumerate(agent.path):
            try:
                validate_next_step_valid(agent.id, state.cell, agent.path[i+1].cell, agent.motion_equation, raise_error)
            except IndexError:
                continue


def validate_paths(grid: grid2d.Grid2D, agent_repo: agent_repository.AgentRepository,
                   raise_error=True):

    validate_agents_and_map_collusion(grid, agent_repo, raise_error)
    validate_motion_equation(agent_repo, raise_error)

