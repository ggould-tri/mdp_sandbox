from mdp import NaiveMdp
from enum import enum

class BernoulliBandit(NaiveMdp):
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
