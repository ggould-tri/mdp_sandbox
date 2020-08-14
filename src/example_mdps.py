from mdp import NaiveMdp
from enum import Enum
import random


class BernoulliBandit(NaiveMdp):
    """The classic one-state random-reward problem."""
    ONLY_STATE = 0

    def __init__(self, arm_payoff_probabilities):
        self.arm_payoff_probabilities = arm_payoff_probabilities

    def actions(self, cur_state):
        return list(range(len(self.arm_payoff_probabilities)))

    def next_state(self, cur_state, action):
        assert cur_state == ONLY_STATE
        return ONLY_STATE

    def reward(self, cur_state, action, next_state):
        return self.arm_payoff_probabilities[action]


class GridWorld(NaiveMdp):
    """Maze-solving as an MDP"""
    def __init__(self, width=4, height=3, walls=None, rewards=None):
        self.width = width
        self.height = height
        self.walls = walls or []
        self.rewards = rewards or {(width - 1, height - 1): 1}

    class Actions(Enum):
        NORTH = (0, 1)
        EAST = (1, 0)
        SOUTH = (0, -1)
        WEST = (-1, 0)

        def prettychar(self):
            return {self.NORTH: '^', self.EAST: '>',
                    self.SOUTH: 'v', self.WEST: '<'}[self]

    def actions(self, _):
        return list(self.Actions)

    def next_state_probabilities(self, cur_state, action):
        x, y = cur_state
        next_state = (x + action.value[0], y + action.value[1])
        if (next_state in self.walls
                or not (0 <= next_state[0] < self.width)
                or not (0 <= next_state[1] < self.height)):
            return {cur_state: 1}
        return {next_state: 1}

    def reward(self, cur_state, action, next_state):
        if next_state in self.rewards:
            return {self.rewards[next_state]: 1}
        return {0: 1}

    def draw(self, draw_policy=None):
        result= ""
        for y in range(self.height-1, -1, -1):
            for x in range(0, self.width):
                if (x, y) in self.walls:
                    result += "#"
                elif (x, y) in self.rewards:
                    result += "+" if self.rewards[(x, y)] > 0 else "-"
                elif draw_policy:
                    action = draw_policy.most_likely_action((x, y))
                    result += action.prettychar()
                else:
                    result += "."
            result += "\n"
        return result


def random_grid_world(rnd=random):
    width = rnd.randrange(3, 8)
    height = rnd.randrange(3, 8)
    mini_area = int(width * height / 3)
    walls = {(rnd.randrange(width), rnd.randrange(height))
             for _ in range(rnd.randrange(mini_area))}
    rewards = dict()
    rewards.update({(rnd.randrange(width), rnd.randrange(height)): -1
                    for _ in range(rnd.randrange(mini_area))})
    rewards.update({(rnd.randrange(width), rnd.randrange(height)): 1
                    for _ in range(rnd.randrange(mini_area))})
    return GridWorld(width, height, walls, rewards)
