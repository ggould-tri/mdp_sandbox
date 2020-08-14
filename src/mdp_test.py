import random
import unittest

from mdp import NaiveMdp, NaiveMdpPolicy, Rollout

class MdpTest(unittest.TestCase):
    class CoinFlipMdp(NaiveMdp):
        def actions(self, state):
            return {0, 1}
        def next_state_probabilities(self, cur_state, action):
            return {cur_state: 0.5, (cur_state + action): 0.5}
        def reward(self, cur_state, action, next_state):
            return {next_state: 1}

    class FlipUntilFifty(NaiveMdpPolicy):
        def action_probabilities(self, cur_state):
            return ({1:1} if cur_state < 50 else {0:1})

    def test_rollout(self):
        m = self.CoinFlipMdp()
        r = Rollout(m, 0, random.Random(0))
        for i in range(100):
            r.step(1, verbose=False)
        assert 35 < r.state < 65, r.state

    def test_policy(self):
        m = self.CoinFlipMdp()
        p = self.FlipUntilFifty()
        r = Rollout(m, 0, random.Random(0))
        for i in range(1000):
            r.policy_step(p, verbose=False)
        assert r.state == 50, r.state


if __name__ == "__main__":
    unittest.main()
