from enum import Enum


class StartPolicy(Enum):
    STAY_AT_START = 'Stay at Start'
    APPEAR_AT_START = 'Appear at Start'


class GoalPolicy(Enum):
    STAY_AT_GOAL = 'Stay at Goal'
    DISAPPEAR_AT_GOAL = 'Disappear at Goal'


class MotionEquation(Enum):
    MOVE_4_DIRECTIONS = '4'
    MOVE_5_DIRECTIONS = '5'
    MOVE_8_DIRECTIONS = '8'
    MOVE_9_DIRECTIONS = '9'


class Agent:
    agent_num = 0
    must_reach_target: int
    start_policy: StartPolicy
    goal_policy: GoalPolicy
    motion_equation : MotionEquation
    is_adversarial: bool
    damage_steps_budget: int
    step_size: int

    # think if the following are needed
    agent_route = []
    x = 0
    y = 0
    step = 0

    def __init__(self, num, must_reach_target, start_policy, goal_policy, motion_equation,
                 is_adversarial, damage_steps_budget):
        self.agent_num = num
        self.must_reach_target = must_reach_target
        self.start_policy = start_policy
        self.goal_policy = goal_policy
        self.motion_equation = motion_equation
        self.is_adversarial = is_adversarial
        self.damage_steps_budget = damage_steps_budget
