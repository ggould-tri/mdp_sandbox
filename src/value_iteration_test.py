import unittest

import value_iteration
from example_mdps import GridWorld, random_grid_world

class ValueIterationTest(unittest.TestCase):
    def test_smoke_test(self):
        """Visit most lines of most of the functions, kinda sorta."""
        gw = random_grid_world()
        print(gw.draw())
        vi = value_iteration.ValueIteration(gw, (0, 0))
        print(gw.draw(vi))
        try:
            vi.stabilize()
        except RuntimeError:
            pass
        print(gw.draw(vi))


if __name__ == "__main__":
    unittest.main()
