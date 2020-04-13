from scipy.spatial.distance import cityblock

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

        

class RewardShapedFrozenLake:

    def __init__(self, frozenlake, gamma=0.999):
        self.env = frozenlake
        self.gamma = gamma

        self.goal_coords = self.env.rows - 1, self.env.cols - 1
        self.max_manhattan_distance = self.manhattan_distance_to_goal(0, 0)


    def manhattan_distance_to_goal(self, x, y):
        return cityblock([x, y], self.goal_coords)

    def get_state_potential(self, x, y):
        return (self.max_manhattan_distance - self.manhattan_distance_to_goal(x, y)) / self.max_manhattan_distance

    def step(self, action):
        s = self.env.pos
        x = s % self.env.cols
        y = s // self.env.rows



        s_p, r, t = self.env.step(action)
        x_p = s_p % self.env.cols
        y_p = s_p // self.env.rows

        this_state_potential = self.get_state_potential(x, y)

        if t and r == 0: # Hole. Don't give manhattan reward
            next_state_potential = 0
        else:
            next_state_potential = self.get_state_potential(x_p, y_p)

        potential = self.gamma * next_state_potential - this_state_potential
        modified_reward = r + potential

        return s_p, modified_reward, t

    def __getattr__(self, attr):
        # Should only be invoked if attr cannot be found in usual way, so should ignore `step`
        return getattr(self.env, attr)
