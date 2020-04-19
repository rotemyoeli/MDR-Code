import logging

from . import adversarial_agent_planner
from . import grid2d
from .adversarial_agent_planner import get_makespan, AgentsType, RobustPathMode, NotFoundError

LOG = logging.getLogger(__name__)


class MdrState(adversarial_agent_planner.AAPState):
    def g(self):
        return -1 * get_makespan(self.paths, self.agents)

    def f(self):
        cost_without_interruptions = self.g()
        return cost_without_interruptions + -1 * sum(int(not a.is_adversarial) for a in self.agents) * self.ds
        # return cost_without_interruptions + -1 * len(self.agents) * self.ds

    @property
    def is_goal(self):
        return self.ds == 0


class MaxDamageRouteFinder(adversarial_agent_planner.AdversarialAgentPlanner):
    def __init__(self,
                 grid: grid2d.Grid2D,
                 agents: AgentsType,
                 searcher_class,
                 searcher_kwargs,
                 robust_mode: RobustPathMode = RobustPathMode.DISABLE,
                 put_adv_agents_in_front=True,
                 robust_radius=None):
        super(MaxDamageRouteFinder, self).__init__(
            grid,
            agents,
            searcher_class,
            searcher_kwargs,
            MdrState,
            put_adv_agents_in_front,
            robust_mode,
            robust_radius
        )
