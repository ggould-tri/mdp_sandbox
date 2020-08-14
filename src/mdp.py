from abc import ABC, abstractmethod
import random


class NaiveMdp(ABC):
    """The minimal (most general) definition of an MDP"""

    @abstractmethod
    def actions(self, cur_state):
        """@return actions allowable from state @p cur_state."""
        pass

    @abstractmethod
    def next_state_probabilities(self, cur_state, action):
        """For a given state-action pair, @return a dict {state->prob} of
        next states and their probabilities."""
        pass

    @abstractmethod
    def reward(self, cur_state, action, next_state):
        """For a given state transition state --action-> state, @return a dict
        {reward->prob} of the eward for that transition."""
        pass


class NaiveMdpPolicy(ABC):
    """The minimal (most general) definition of a decision policy."""
    @abstractmethod
    def action_probabilities(self, cur_state):
        """For a given state, @return a dict {action->prob} of
        next actions and their probabilities."""
        pass


class Rollout:
    def __init__(self, mdp, initial_state, randomness=random):
        self.state = initial_state
        self.mdp = mdp
        self.randomness = randomness
        self.total_reward = 0

    def _weighted_choice(self, outcome_dict):
        return self.randomness.choices(
            population=list(outcome_dict),
            weights=list(outcome_dict.values()))[0]

    def step(self, action, verbose=True):
        assert action in self.mdp.actions(self.state)
        states = self.mdp.next_state_probabilities(self.state, action)
        next_state = self._weighted_choice(states)
        rewards = self.mdp.reward(self.state, action, next_state)
        reward = self._weighted_choice(rewards)
        self.total_reward += reward
        if verbose:
            print(f"{self.state} --{action}-> {next_state}  :: "
                f"{self.total_reward} + {reward} = "
                f"{self.total_reward + reward}")
        self.state = next_state

    def policy_step(self, policy, verbose=True):
        actions = policy.action_probabilities(self.state)
        action = self._weighted_choice(actions)
        self.step(action, verbose)
