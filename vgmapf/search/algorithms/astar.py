import queue
from typing import Tuple, List

from . import algo_utils
from .best_first_search import BestFirstSearch
from vgmapf.search import base_state
from vgmapf.search.base_algorithm import SearchAlgorithm


class AStar(BestFirstSearch):
    def __init__(self, start_state, observer, h_func):
        self.h_func = h_func
        super(AStar, self).__init__(start_state, observer)

    def _get_node_f(self, node, node_g):
        return node_g + self.h_func(node)


Searcher = AStar
