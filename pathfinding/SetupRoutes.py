#############################################################################
# import packages
##############################################################################
import random
import numpy as np
import os
from tkinter import *
import itertools as itert
from pathfinding.Utils import utils
import logging
import pandas as pd

from pathfinding.Core.diagonal_movement import DiagonalMovement
from pathfinding.Core.grid import Grid
from pathfinding.SearchAlgorithms.a_star import AStarFinder
from pathfinding.SearchAlgorithms.dijkstra import DijkstraFinder
###################################################################################
# Main Setup Route
###################################################################################
from vgmapf.problems.mapf import paths_serializer
from vgmapf.utils import benchmark_utils

LOG = logging.getLogger(__name__)

def create_routes(map_file_name, data_folder, agents_data, num_of_routes, starts_arr=None, goals_arr=None):
    num_of_agents = len(agents_data)

    grid = load_map_to_grid(map_file_name)

    os.chdir(data_folder)
    path = data_folder + "\\Routes Output"
    if not os.path.exists(path):
        os.makedirs(path)
    os.chdir(path)

    if not starts_arr or not goals_arr:
        starts_arr, goals_arr = utils.get_start_and_goal_positions(grid, num_of_agents)

    for reset_grid in range(0, len(starts_arr)):
        grid[starts_arr[reset_grid]] = 1
        grid[goals_arr[reset_grid]] = 1

    #########################################
    # build unique agents order array
    #########################################

    tmp_str = ''
    for build_str in range(0, num_of_agents):
        tmp_str = tmp_str + str(build_str)

    if 1 == num_of_routes:
        order_arr = [tuple(tmp_str)]
    else:
        order_arr = list(itert.permutations(tmp_str, num_of_agents))
    random.shuffle(order_arr)


    ##################################################
    # Set here the number of unique routes
    ##################################################
    for current_route in range(0, min(num_of_routes, len(order_arr))):
        with benchmark_utils.time_it(f'Building route #{current_route}'):
            routes = [[] for i in range(num_of_agents)]
            current_agents_order = list(map(int, order_arr[current_route]))
            print("current_agents_order: " + str(current_agents_order))

            # build each agent's route
            for agent_index in range(0, num_of_agents):
                #agent_num = current_agents_order[agent_index]
                agent_num = agent_index # temp fixed order for debug
                ############################
                # Reset the grid
                ############################
                curr_grid = Grid(matrix=grid)
                start = curr_grid.node(starts_arr[agent_num][1], starts_arr[agent_num][0])
                goal = curr_grid.node(goals_arr[agent_num][1], goals_arr[agent_num][0])

                # TODO create agent's finder with the correct motion equation
                finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
                # finder = DijkstraFinder(diagonal_movement=DiagonalMovement.always)
                path, runs = finder.find_path(start, goal, curr_grid, routes, agent_num, agents_data)
                # routes.append(path)
                routes[agent_num] = path
                LOG.info("\n\n************** Agent #" + str(agent_num) + "**************")
                LOG.info('operations:' + str(runs) +  'path length:' + str(len(path)) + 'path cost: ' + str(path[-1].g))
                print(curr_grid.grid_str(path=path, start=start, end=goal))
                LOG.info( 'Path: ' + ' '.join(str(p) for p in path))

        ##################################################
        # Save routes to csv file
        ##################################################
        file_name_csv = "Route-" + str(current_route+1) + '_Agents-' + str(num_of_agents) + '.csv'
        paths_serializer.dump(file_name_csv, paths=routes)

        # df_res = pd.DataFrame(routes)
        # # TODO add metadata to the file / add informative name with all agents data
        # file_name_csv = "Route-" + str(current_route+1) + '_Agents-' + str(num_of_agents) + '.csv'
        # print(file_name_csv)
        # df = pd.DataFrame(routes)
        # df.to_csv(file_name_csv, index=False, header=False)

    path = '..'
    os.chdir(path)


def load_map_to_grid(map_file_name):
    # load the map file into numpy array
    with open(map_file_name, 'rt') as infile:
        grid1 = np.array([list(line.strip()) for line in infile.readlines()])
    print('Grid shape', grid1.shape)
    grid1[grid1 == '@'] = 0  # object on the map
    grid1[grid1 == 'T'] = 0  # object on the map
    grid1[grid1 == '.'] = 1  # free on map
    grid = np.array(grid1.astype(np.int))
    return grid
