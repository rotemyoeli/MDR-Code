import sys
from dataclasses import dataclass
from typing import List, Tuple, Dict

from vgmapf.problems.mapf.mdr_finder import RobustPathMode
from .problems.mapf import agent
import yaml


@dataclass
class RunConfig:
    map_file_name: str

    agents: List[Dict]

    start: Tuple[int, int] = None
    end: Tuple[int, int] = None

    permutations: int = 1

    robust_route: RobustPathMode = RobustPathMode.DISABLE

    min_start_end_distance: int = 0
    max_start_end_distance: int = sys.maxsize


def _safe_tuple(x):
    if x is None:
        return None
    return tuple(x)


def load(yaml_file_name) -> RunConfig:
    with open(yaml_file_name, 'r') as f:
        d = yaml.load(f, Loader=yaml.FullLoader)

    d['start'] = _safe_tuple(d.get('start'))
    d['end'] = _safe_tuple(d.get('end'))

    rc = RunConfig(
        **d
    )

    for a in rc.agents:
        a['start_policy'] = agent.StartPolicy.STAY_AT_START if a.get('start_policy') is None else agent.StartPolicy(a.get('start_policy'))
        a['goal_policy'] = agent.GoalPolicy.STAY_AT_GOAL if a.get('goal_policy') is None else agent.GoalPolicy(a['goal_policy'])
        a['motion_equation'] = agent.MotionEquation.MOVE_5_DIRECTIONS if a.get('motion_equation') is None else agent.MotionEquation(a['motion_equation'])

        a['start_cell'] = _safe_tuple(a.get('start_cell'))
        a['goal_cell'] = _safe_tuple(a.get('goal_cell'))

    return rc
