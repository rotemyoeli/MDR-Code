import heapq
import numpy

from . import grid2d
from . import agent

def dijkstra(grid: grid2d.Grid2D, source, motion_equation: agent.MotionEquation):
    dist_to_source = dict()
    dist_to_source[source] = 0

    oheap = []
    closed = set()
    heapq.heappush(oheap, (0, source))

    while oheap:
        (g, pos) = heapq.heappop(oheap)
        closed.add(pos)

        for neighbor_distance, neighbor_cell in grid.get_accessible_neighbors(pos, motion_equation=motion_equation):
            new_g = g + neighbor_distance

            if neighbor_cell in closed:
                continue

            if neighbor_cell in dist_to_source:
                # If existing path is better
                if dist_to_source[neighbor_cell] <= new_g:
                    continue

            dist_to_source[neighbor_cell] = new_g
            heapq.heappush(oheap, (new_g, neighbor_cell))
    return dist_to_source