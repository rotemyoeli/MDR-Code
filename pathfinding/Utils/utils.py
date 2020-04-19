
import heapq
import logging
import random
import numpy as np
import heapq
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import openpyxl as openpyxl
import csv
import ast

from pathfinding.Core.node import Node

LOG = logging.getLogger(__name__)

'''
Check if pos is a legal position in the map
'''
def is_legal(grid, pos):
    if 0 <= pos[0] < grid.shape[0]:
        if 0 <= pos[1] < grid.shape[1]:
            if grid[pos[0]][pos[1]] == 1:
                return False # array bound 1 on map
        else:
            return False # array bound y walls
    else:
        return False # array bound x walls
    return True

'''
Return a dictionary that maps every location to the distance from the given start
'''
def dijkstra(grid, source, operators):
    dist_to_source = dict()
    dist_to_source[source]=0

    oheap = []
    closed = set()
    heapq.heappush(oheap, (0, source))

    while oheap:
        (g, pos) = heapq.heappop(oheap)
        closed.add(pos)

        for i, j, cost in operators:
            neighbor = (pos[0] + i, pos[1] + j)
            new_g = g+cost

            if is_legal(grid, neighbor)==False:
                continue

            if neighbor in closed:
                continue

            if neighbor in dist_to_source:
                # If existing path is better
                if dist_to_source[neighbor]<=new_g:
                    continue
                # TODO: In the future,

            dist_to_source[neighbor]=new_g
            heapq.heappush(oheap,(new_g, neighbor))
    return dist_to_source



def find_free_place(grid_tmp, current_agent_pos, neighbors):

    current_not_set = 1
    while current_not_set == 1:
        for i, j in neighbors:
            neighbor = (current_agent_pos[0] + i, current_agent_pos[1] + j)
            if (neighbor[0] > grid_tmp.shape[0] or neighbor[1] > grid_tmp.shape[1]) or grid_tmp[neighbor] == 0:
                continue
            else:
                grid_tmp[neighbor] = 0
                return grid_tmp, neighbor
        for i, j in neighbors:
            neighbor = (current_agent_pos[0] + i, current_agent_pos[1] + j)
            grid_tmp, neighbor = find_free_place(grid_tmp, neighbor, neighbors)
            return grid_tmp, neighbor



##############################################################################
# Valid start according to the robust level
##############################################################################
# currently not used
def get_start_approval(array, start_pos, agent_no, route, robust_dist):

    radius = get_radius_nodes(array, start_pos[0], robust_dist)
    ##Check for collision
    for a in range(0, agent_no):

        for robust_check in range(0, robust_dist):

            if len(route[a]) > start_pos[1]:

                for radius_check in range(0, len(radius)):
                    if route[a][start_pos[1]][0] == radius[radius_check] and a != agent_no:
                        LOG.debug('a -', a, 'agent no -', agent_no, route[a][start_pos[1]], radius[radius_check])

                        return False

    return True


def in_start_state(current, route, stp, y):
    is_in_start = False
    for check_start in range(0, stp):
        if route[y][check_start][0] == current[0]:
            is_in_start = True
        else:
            return False
    return is_in_start

def _compute_makespan(routes):
    return max([len(route) for route in routes])

##############################################################################
# heuristic function for heuristic_interrupt path scoring
##############################################################################
'''
search_node contains position of all agent, allowed abnormal moves, current time step
routes contains the planned routes for all agents [ including the path they passed so far ]
'''
def f_interrupt(self, search_node, routes):

    cost_without_interrupts = MDR._compute_makespan(routes)

    # An overestimate of the cost added by the abnormal moves
    remaining_abnormal_moves = search_node.k
    num_of_agents = len(search_node.pos)

    # The idea is that the maximal addition for each abnormal move is that it will delay all agents by one time step.
    return -1*(cost_without_interrupts + num_of_agents * remaining_abnormal_moves)  # TODO: CHANGE 2

    '''
     route = list containing the planned route for each agent 
    '''

def is_goal(self, node, route):

    for x in range(0, len(node.pos)):
        if node.pos[x] is not route[x][-1][0] and node.pos[x] is not None:
            return False
    return True


##############################################################################
# get start and goal for all agents (not the lead)
##############################################################################

def get_start_and_goal_positions(grid_tmp, agent_number):
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    new_starts = []
    new_goals = []
    x_size = grid_tmp.shape[0]
    y_size = grid_tmp.shape[1]

    # set the first agent S&G positions on valid places on the grid
    start_is_set = 0
    goal_is_set = 0
    while start_is_set == 0:
        new_starts.append((random.sample(range(0, x_size), 1)[0], random.sample(range(0, y_size), 1)[0]))
        if grid_tmp[new_starts[0]] == 0:
            new_starts = []
            continue
        else:
            grid_tmp[new_starts[0]] = 0
            start_is_set = 1
            break

    while goal_is_set == 0:
        new_goals.append((random.sample(range(0, x_size), 1)[0], random.sample(range(0, y_size), 1)[0]))
        if grid_tmp[new_goals[0]] == 0:
            new_goals = []
            continue
        else:
            grid_tmp[new_goals[0]] = 0
            goal_is_set = 1
            break


    # set all S&G according the leader
    for num_of_SG in range(1, agent_number):
        # Find start position for the next agent
        current_agent_pos = new_starts[0]
        grid_tmp, neighbor = find_free_place(grid_tmp, current_agent_pos, neighbors)
        new_starts.append(neighbor)

        # Find goal position for the next agent
        current_agent_pos = new_goals[0]
        grid_tmp, neighbor = find_free_place(grid_tmp, current_agent_pos, neighbors)
        new_goals.append(neighbor)

    ########################################
    # Fixed routes for testing
    ########################################
    #new_starts = [(10, 5), (11, 5), (12, 5), (10, 4), (11, 5)]
    #new_goals = [(12, 10), (10, 10), (11, 10), (11, 11), (12, 11)]

    return new_starts, new_goals


##############################################################################
# Return the radius - robust around the agent no. 0 - The adversarial
# the radius is a list of nodes that risk agent no. 0, meaning all nodes
# from which another agent can reach agent no. 0 using "robust_param" steps
##############################################################################
def get_radius_nodes(grid, node, robust_param, all_agents_routes, current_agent_no):
    robust_radius = []

    if len(all_agents_routes) == 0:
        return []

    agent0_route = all_agents_routes[0]

    if robust_param == 0 and node.step <= len(agent0_route[0]):
        return all_agents_routes[current_agent_no][node.step]  # Irit - why?

    # verify Agent0 exists on the map during this step
    # this check could be wrong if agent0 goal policy is "stay at goal" - the length of the path can be shorter than
    # other agents steps but they still need to verify they keep the radius from him
    if node.step >= len(agent0_route):
        return []

    left_up = ((agent0_route[node.step][0][0] - (2 * robust_param)), agent0_route[node.step][0][1] - (2 * robust_param))
    length = (robust_param * 4) + 1

    for step_right in range(0, length):
        for step_down in range(0, length):
            neighbor = (left_up[0] + step_right, left_up[1] + step_down)
            LOG.debug("neighbor=( " + str(neighbor[0]) + ", " + str(neighbor[1]) + ")")
            if neighbor[0] < len(grid.nodes) \
                    and neighbor[0] < len(grid.nodes[neighbor[0]]) \
                    and grid.nodes[neighbor[0]][neighbor[1]].is_walkable:

                robust_radius.append(neighbor)

    return robust_radius


def get_dangerous_nodes(grid, curr_agent_no, step, all_agents_routes, agents_data):
    # TODO block the radius by walls
    # TODO think what to do with "stay at goal" - maybe append this stay steps to the end of all routes

    # for each existing agent's route:
    #   avoid collision: add the next step of that agent
    #   if the other agent is adversarial-> add to its radius according to next step and its damage steps
    #   if I'm adversarial -> make sure my next step keeps the other agent safe from me:
    #       verify the next D steps do not enter my radius (D= my damage steps)
    dangerous_nodes = []

    for other_agent_num, other_agent_route in enumerate(all_agents_routes):
        if len(other_agent_route) == 0:
            continue

        if other_agent_num == curr_agent_no:
            continue

        # other agent finished his route
        if step >= len(other_agent_route):
            continue

        # avoid collision
        dangerous_nodes.append(other_agent_route[step])

        # stay away from adversarial agents
        if agents_data[other_agent_num].is_adversarial:
            # TODO maybe add a check that the other agent exists in the next step
            # TODO think if we need to check the radius from the current position or the next (the step)
            dangerous_nodes += (get_dangerous_square_nodes(grid, other_agent_route[step],
                                                           agents_data[other_agent_num].damage_steps_budget))

        # don't risk others - don't enter their safe square (on the next step)
        if agents_data[curr_agent_no].is_adversarial and step+1 < len(other_agent_route):
            dangerous_nodes += (get_dangerous_square_nodes(grid, other_agent_route[step+1],
                                                           agents_data[curr_agent_no].damage_steps_budget))

    LOG.debug("AgentNum: " + str(curr_agent_no) + " Step: " + str(step) + " dangerous_nodes " + ' '.join(str(d) for d in dangerous_nodes))
    return dangerous_nodes


def get_dangerous_square_nodes(grid, node, damage_steps):
    # calculate the (2*damage_steps + 1)^2 square around node
    dangerous_square_nodes = []
    upper_left_corner_x = node.x - damage_steps
    upper_left_corner_y = node.y - damage_steps

    length = 2*damage_steps + 1

    for i_x in range(length):
        for i_y in range(length):
            if grid.is_walkable(upper_left_corner_x + i_x, upper_left_corner_y + i_y):
                dangerous_square_nodes.append(Node(x=upper_left_corner_x + i_x, y=upper_left_corner_y + i_y))

    return dangerous_square_nodes

