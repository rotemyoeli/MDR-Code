from abc import ABC, abstractmethod
from typing import List, Tuple


class State(ABC):
    step = 0

    @abstractmethod
    def expand(self) -> List[Tuple[int, "State"]]:
        """
        Returns list of (distance, state) that are directly reachable from self
        """
        pass

    @property
    @abstractmethod
    def is_goal(self):
        pass

    @abstractmethod
    def _internal_state(self):
        """
        A hashable/mutable representation of self
        """
        pass

    def g(self):
        return None

    def f(self):
        return None

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self._internal_state() == other._internal_state()

        return self._internal_state() == other

    def __hash__(self):
        return hash(self._internal_state())

    def __lt__(self, other):
        # noinspection PyProtectedMember
        return self._internal_state() < other._internal_state()
