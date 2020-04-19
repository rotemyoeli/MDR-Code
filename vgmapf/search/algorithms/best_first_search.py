from abc import abstractmethod
from typing import Tuple, List

from . import algo_utils
from vgmapf.search import base_state
from vgmapf.search import base_algorithm


class BestFirstSearch(base_algorithm.SearchAlgorithm):
    def search(self) -> Tuple[List[base_state.State], int]:
        prepo = algo_utils.NodeRepository()
        open_set = algo_utils.HashedPriorityQueue(prepo.get_node_f)

        prepo.add_root(self._start_state)
        prepo.set_node_f(self._start_state, self._get_node_f(self._start_state, node_g=0))
        open_set.put(self._start_state)

        while not open_set.empty():
            cur_node, f = open_set.get()  # type: (base_state.State, int)

            self._observer.onStateExpanded(cur_node)

            if cur_node.is_goal:
                return prepo.backtrack(cur_node), prepo.get_cost(cur_node)

            for d, new_node in cur_node.expand():
                new_g = prepo.get_cost(cur_node) + d
                if new_node not in prepo or new_g < prepo.get_cost(new_node):
                    prepo.add(new_node, cur_node, d)
                    new_f = self._get_node_f(new_node, new_g)
                    prepo.set_node_f(new_node, new_f)
                    # if new_node in open_set:
                    #     open_set.remove(new_node)
                    open_set.put(new_node)

        raise base_algorithm.NotFoundError()

    @abstractmethod
    def _get_node_f(self, node, node_g):
        pass

