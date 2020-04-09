class HuntingMDPWrapper:
    """A wrapper to make hunting MDP compatible with QLearner"""


    def __init__(self, huntingmdp):
        self.env = huntingmdp


    def step(self, action):
        # Convert state from index to state
        next_state, reward, terminal = self.env.step(action)

        next_state_ix = self.env.state_to_idx[next_state]
        return next_state_ix, reward, terminal

    def reset(self):
        s = self.env.reset()
        return self.env.state_to_idx[s]


    def __getattr__(self, attr):
        # Should only be invoked if attr cannot be found in usual way, so should ignore `step`
        return getattr(self.env, attr)

        

class FrozenLakeWrapper:

    def __init__(self, frozenlake):
        self.env = frozenlake

    def __getattr__(self, attr):
        return getattr(self.env, attr)

