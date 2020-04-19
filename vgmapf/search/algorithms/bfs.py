from typing import List, Tuple

from vgmapf.search import base_state
from vgmapf.search.base_algorithm import SearchAlgorithm
from . import algo_utils


class BFS(SearchAlgorithm):
    def search(self) -> Tuple[List[base_state.State], int]:
        self._start_state.g = 0
        self._start_state.parent = None

        prepo = algo_utils.NodeRepository()

        prepo.add_root(self._start_state)
        frontier = [self._start_state]

        while frontier:
            node = frontier.pop(0)
            self._observer.onStateExpanded(node)
            if node.is_goal:
                path = prepo.backtrack(node)
                cost = prepo.get_cost(node)
                return path, cost

            for d, n in node.expand():
                if n in prepo:
                    continue
                prepo.add(n, node, d)
                frontier.append(n)


Searcher = BFS
