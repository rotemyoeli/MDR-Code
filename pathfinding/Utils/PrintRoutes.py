# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 08:05:00 2018

@author: rotem
"""
#############################################################################
# import packages
##############################################################################
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import csv
import ast
import pylab as pl

def print_route(map_file, route_file, show_step_num):
    # load the map file into numpy array
    with open(map_file, 'rt') as infile:
        grid1 = np.array([list(line.strip()) for line in infile.readlines()])
    print('Grid shape', grid1.shape)

    grid1[grid1 == '@'] = 1 #object on the map
    grid1[grid1 == 'T'] = 1 #object on the map
    grid1[grid1 == '.'] = 0 #free on map

    grid = np.array(grid1.astype(np.int))

    # Plot the map
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(grid, cmap=plt.cm.gray)

    all_routes = []

    with open(route_file, 'rt') as csvfile:
        csv_reader = csv.reader(csvfile)
        # extracting each data row one by one
        for row in csv_reader:
            tmp_route = []
            for step in row:
                if step:
                    tmp_route.append(ast.literal_eval(step))
            all_routes.append(tmp_route)

    ##############################################################################
    # plot the path
    ##############################################################################
    colors = ['white', 'green', 'blue', 'yellow', 'red', 'cyan', 'magenta', 'purple', 'brown', 'pink', 'gray', 'olive']
    color_i = 0
    for curr_route in all_routes:
        if len(curr_route) == 0:
            continue

        start = curr_route[0][0]
        goal = curr_route[len(curr_route)-1][0]

        # ax.scatter(start[1], start[0], marker="^", color=colors[color_i], s=50)
        # ax.scatter(goal[1], goal[0], marker="*", color=colors[color_i], s=50)
        ax.scatter(start[0], start[1], marker="^", color=colors[color_i], s=50)
        ax.scatter(goal[0], goal[1], marker="*", color=colors[color_i], s=50)

        # extract x and y coordinates from route list
        x_coords = []
        y_coords = []

        for ((x, y), step) in curr_route:
            if show_step_num:
                pl.text(x, y, str(step), color=colors[color_i], fontsize=11)
            x_coords.append(x)
            y_coords.append(y)

        pl.margins(0.1)
        # ax.plot(y_coords, x_coords, "-0", color=colors[color_i])
        ax.plot(x_coords, y_coords, "-0", color=colors[color_i])

        color_i += 1
        if color_i >= len(colors):
            color_i = 0

    plt.grid(True)
    plt.show()
