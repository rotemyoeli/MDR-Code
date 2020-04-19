from .best_first_search import BestFirstSearch


class UniformCostSearch(BestFirstSearch):
    def _get_node_f(self, node, node_g):
        return node_g


Searcher = UniformCostSearch
