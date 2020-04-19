import logging
import heapq
import queue

from typing import Dict, Tuple

LOG = logging.getLogger(__name__)

class NodeRepository:
    def __init__(self):
        self.parents: Dict[object, Tuple[object, int]] = dict()
        self.gs: Dict[object, int] = dict()
        self.fs: Dict[object, int] = dict()

    def add_root(self, node):
        self.parents[node] = None, 0
        self.gs[node] = 0

    def add(self, node, parent=None, distance=None):
        if parent is not None:
            self.parents[node] = (parent, distance)
            if distance is not None:
                self.gs[node] = self.gs[parent] + distance
        else:
            self.gs[node] = None

    def set_node_f(self, node, f):
        self.fs[node] = f

    def get_node_f(self, node):
        return self.fs[node]

    def remove(self, node):
        del self.parents[node]
        del self.gs[node]
        del self.fs[node]

    def backtrack(self, node) -> list:
        path = []
        cur = node
        while cur:
            path.insert(0, cur)
            cur, d = self.parents[cur]

        return path

    def get_cost(self, node) -> int:
        return self.gs[node]

    def __contains__(self, node):
        return node in self.gs


class HashedPriorityQueueError(Exception):
    pass


class NodeAlreadyExistsError(HashedPriorityQueueError):
    pass


class SlowHashedPriorityQueue:
    def __init__(self):
        self.nodes = set()

    def put(self, node):
        """
        We assume (and verify) that the same node isn't put into the HashedPriorityQueue again if it already exists in it,
        as this shouldn't happen during astar.
        If putting same node with different priorities is required, then a more complex data structure needs to be impelmented
        since this simple composition doesn't allow implementing this use case efficiently.
        """
        self.nodes.add(node)

    def get(self, f_func):
        node = min(self.nodes, key=lambda n: f_func(n))
        f = f_func(node)
        self.nodes.remove(node)
        return node, f

    def remove(self, node):
        self.nodes.remove(node)

    def empty(self):
        return True if not self.nodes else False

    def __contains__(self, node):
        return node in self.nodes


class PriorityDict(dict):
    """Dictionary that can be used as a priority queue.

    Keys of the dictionary are items to be put into the queue, and values
    are their respective priorities. All dictionary methods work as expected.
    The advantage over a standard heapq-based priority queue is
    that priorities of items can be efficiently updated (amortized O(1))
    using code as 'thedict[item] = new_priority.'

    The 'smallest' method can be used to return the object with lowest
    priority, and 'pop_smallest' also removes it.

    The 'sorted_iter' method provides a destructive sorted iterator.
    """

    def __init__(self, *args, **kwargs):
        super(PriorityDict, self).__init__(*args, **kwargs)
        self._rebuild_heap()

    def _rebuild_heap(self):
        self._heap = [(v, k) for k, v in self.items()]
        heapq.heapify(self._heap)

    def smallest(self):
        """Return the item with the lowest priority.

        Raises IndexError if the object is empty.
        """

        heap = self._heap
        v, k = heap[0]
        while k not in self or self[k] != v:
            heapq.heappop(heap)
            v, k = heap[0]
        return k

    def pop_smallest(self):
        """Return the item with the lowest priority and remove it.

        Raises IndexError if the object is empty.
        """

        heap = self._heap
        v, k = heapq.heappop(heap)
        while k not in self or self[k] != v:
            v, k = heapq.heappop(heap)
        del self[k]
        return k, v

    def __setitem__(self, key, val):
        # We are not going to remove the previous value from the heap,
        # since this would have a cost O(n).

        super(PriorityDict, self).__setitem__(key, val)

        if len(self._heap) < 2 * len(self):
            heapq.heappush(self._heap, (val, key))
        else:
            # When the heap grows larger than 2 * len(self), we rebuild it
            # from scratch to avoid wasting too much memory.
            self._rebuild_heap()

    def setdefault(self, key, val):
        if key not in self:
            self[key] = val
            return val
        return self[key]

    def update(self, *args, **kwargs):
        # Reimplementing dict.update is tricky -- see e.g.
        # http://mail.python.org/pipermail/python-ideas/2007-May/000744.html
        # We just rebuild the heap from scratch after passing to super.

        super(PriorityDict, self).update(*args, **kwargs)
        self._rebuild_heap()

    def sorted_iter(self):
        """Sorted iterator of the priority dictionary items.

        Beware: this will destroy elements as they are returned.
        """

        while self:
            yield self.pop_smallest()


class HybriddPriorityQueue:
    def __init__(self, f_func):
        self.queue = queue.PriorityQueue()
        self.nodes = set()

        self.f_func = f_func

    def put(self, node):
        if node in self.nodes:
            # with benchmark_utils.time_it(f'{node} already in open_list, removing it'):
            self.nodes.remove(node)
            self.queue.queue = [(f, n) for f, n in self.queue.queue if n != node]
            heapq.heapify(self.queue.queue)

        f = self.f_func(node)
        self.queue.put((f, node))
        self.nodes.add(node)

    def get(self):
        f, node = self.queue.get_nowait()
        try:
            self.nodes.remove(node)
            self.queue.queue
        except KeyError:
            pass
        return node, f

    def empty(self):
        return self.queue.empty()

    def __contains__(self, node):
        return node in self.nodes

class HashedPriorityQueue:
    def __init__(self, f_func):
        self.priority_dict = PriorityDict()

        self.f_func = f_func

    def put(self, node):
        f = self.f_func(node)
        self.priority_dict[node] = f

    def get(self):
        node, f = self.priority_dict.pop_smallest()
        return node, f

    def empty(self):
        return False if self.priority_dict else True
