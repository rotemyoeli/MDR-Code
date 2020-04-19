from .best_first_search import BestFirstSearch


class PureHeuristicSearch(BestFirstSearch):
    def __init__(self, start_state, observer, h_func):
        self.h_func = h_func
        super(PureHeuristicSearch, self).__init__(start_state, observer)

    def _get_node_f(self, node, node_g):
        return self.h_func(node)


Searcher = PureHeuristicSearch
