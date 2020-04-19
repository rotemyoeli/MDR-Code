import sys

from abc import ABC, abstractmethod
from typing import List, Tuple

from . import base_state


class NotFoundError(Exception):
    pass


class SearchObserver(ABC):
    @abstractmethod
    def onStateExpanded(self, state):
        pass


class _BlankSearchObserver(SearchObserver):
    def onStateExpanded(self, state):
        pass


class NodeCounter(SearchObserver):
    def __init__(self):
        self.count = 0

    def onStateExpanded(self, state):
        self.count += 1
        if 0 == self.count % 1000:
            sys.stdout.write(f'\rnodes: {self.count}')


class SearchAlgorithm(ABC):
    def __init__(self, start_state: base_state.State, observer: SearchObserver):
        self._start_state = start_state
        if observer is None:
            observer = _BlankSearchObserver()
        self._observer = observer

    @abstractmethod
    def search(self) -> Tuple[List[base_state.State], int]:
        pass
