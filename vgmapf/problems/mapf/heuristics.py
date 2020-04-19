import math

from . import agent

SQRT2 = math.sqrt(2)


def null(s):
    """
    special heuristic for Dijkstra
    return 0, so node.h will always be calculated as 0,
    distance cost (node.f) is calculated only from
    start to current point (node.g)
    """
    return 0


def manhattan_distance(dx, dy):
    return dx + dy


def manhatten(s):
    """manhatten heuristics"""
    dx = abs(s.x - s.gx)
    dy = abs(s.y - s.gy)
    return manhattan_distance(dx, dy)


def euclidean_distance(dx, dy):
    return math.sqrt(dx * dx + dy * dy)


def euclidean(s):
    """euclidean distance heuristics"""
    dx = abs(s.x - s.gx)
    dy = abs(s.y - s.gy)
    return euclidean_distance(dx, dy)


def chebyshev_distance(dx, dy):
    return max(dx, dy)


def chebyshev(s):
    """ Chebyshev distance. """
    dx = abs(s.x - s.gx)
    dy = abs(s.y - s.gy)
    return chebyshev_distance(dx, dy)


def octile_distance(dx, dy):
    f = SQRT2 - 1
    if dx < dy:
        return f * dx + dy
    else:
        return f * dy + dx


def octile(s):
    dx = abs(s.x - s.gx)
    dy = abs(s.y - s.gy)

    return octile_distance(dx, dy)


def get_good_manhatten_like_heuristic(agnt: agent.Agent):
    return octile if agnt.motion_equation in (
    agent.MotionEquation.MOVE_8_DIRECTIONS, agent.MotionEquation.MOVE_9_DIRECTIONS) else manhatten
