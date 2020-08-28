from collections import defaultdict
from copy import deepcopy

from mdp import NaiveMdpPolicy

class ValueIteration(NaiveMdpPolicy):
    """A class containing the state of a value iteration process, which can
    act as a policy using the current value iteration state.

    Value iteration finds a policy via dynamic programming by directly
    interrogating an MDP's transition graph, and is optimal where such
    direct access is possible.
    """
    def __init__(self, mdp, starting_state):
        self.mdp = mdp
        self.starting_state = starting_state
        self.values = defaultdict(lambda: 0)
        self.values[starting_state] = 0

    def stabilize(self,
                  search_iterations=100,
                  refine_iterations=100,
                  discount_factor=0.9):
        """Heuristically attempt to stabilize this policy function."""
        for i in range(search_iterations):
            updated = self.iterate_values(discount_factor=discount_factor)
            if self.values[self.starting_state] != 0:
                break
            if not updated: return
        else:
            raise RuntimeError("No path to reward found after "
                               f"{search_iterations} iterations")
        for j in range(i):
            updated = self.iterate_values(discount_factor=discount_factor)
            if not updated: return

    def iterate_values(self, discount_factor=0.9):
        new_values = defaultdict(lambda: 0)
        for state in self.values.keys():
            action_values = self.action_values(state, discount_factor)
            action_probabilities = self.compute_action_probabilities(
                action_values)
            state_value = 0
            for a, a_probability in action_probabilities.items():
                state_value += a_probability * action_values[a]
            new_values[state] = state_value
        updated = (self.values != new_values)
        self.values = new_values
        return updated

    def action_values(self, cur_state, discount_factor=0.9):
        action_values = {}
        for a in self.mdp.actions(cur_state):
            nexts = self.mdp.next_state_probabilities(cur_state, a)
            total_action_value = 0
            for next_state, probability in nexts.items():
                reward = self.mdp.expected_reward(cur_state, a, next_state)
                # Side effect:  Adds `next_state` to keys of `values`.
                future_value = self.values[next_state] * discount_factor
                total_action_value += probability * (reward + future_value)
            action_values[a] = total_action_value
        return action_values

    def compute_action_probabilities(self, action_values):
        max_value = max(action_values.values())
        max_actions = [k for k, v in action_values.items() if v == max_value]
        return {a: 1/len(max_actions) for a in max_actions}

    ### Policy function:
    def action_probabilities(self, cur_state):
        action_values = self.action_values(cur_state)
        return self.compute_action_probabilities(action_values)
