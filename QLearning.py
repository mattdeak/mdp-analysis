import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict

def choose_random_best_action(q_state_vals):
    """choose_random_best_action

    A utility function for randomly choosing between equal-valued best actions

    Parameters
    ----------

    q_state_vals : A row of a Q table (1-D)

    Returns randomly chosen best action
    -------
    """
    best_ixs = np.where(q_state_vals == q_state_vals.max())[0]
    return np.random.choice(best_ixs)


class ExplorationStrategy(ABC):
    @abstractmethod
    def choose_action(self, state, q_table):
        """Choose an action for the state"""

    def __call__(self, state, q_table):
        # So you can use the class as a function
        return self.choose_action(state, q_table)

class RandomExploration(ExplorationStrategy):

    def choose_action(self, state, q_table):
        return np.random.randint(q_table.shape[1])

class EqualExploration(ExplorationStrategy):

    def __init__(self):
        self.n = {}

    def choose_action(self, state, q_table):
        visit_counts = self.n.get(state)
        if visit_counts is None:
            self.n[state] = {x: 0 for x in range(q_table.shape[1])}

        action = min(self.n[state], key=self.n[state].get)
        self.n[state][action] += 1

        return action

        
class GreedyExploration(ExplorationStrategy):

    def choose_action(self, state, q_table):

        return choose_random_best_action(q_table[state])

class EpsilonGreedyConstant(ExplorationStrategy):
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def choose_action(self, state, q_table):
        rnd = np.random.random()
        if rnd < self.epsilon:
            action = np.random.randint(q_table.shape[1])
        else:
            action = choose_random_best_action(q_table[state])
        return action


class EpsilonGreedyDecay(ExplorationStrategy):
    def __init__(self, epsilon_init=1.0, decay_rate=0.0001, min_epsilon=0.001):
        self.epsilon = epsilon_init
        self.decay_rate = decay_rate
        self.min_epsilon = min_epsilon

    def choose_action(self, state, q_table):
        rnd = np.random.random()
        if rnd < self.epsilon:
            action = np.random.randint(q_table.shape[1])
        else:
            action = choose_random_best_action(q_table[state])

        # decay epsilon
        if self.epsilon > self.min_epsilon:
            self.epsilon = max(self.epsilon * (1 - self.decay_rate), self.min_epsilon)

        return action


class EpsilonGreedyStateBasedDecay(ExplorationStrategy):
    def __init__(self):
        self.state_counter = defaultdict(int)

    def choose_action(self, state, q_table):
        # Need to make state hashable
        if isinstance(state, list):
            state = tuple(state)

        self.state_counter[state] += 1
        epsilon = 1 / self.state_counter[state]

        rnd = np.random.random()
        if rnd < epsilon:
            action = np.random.randint(q_table.shape[1])
        else:
            action = choose_random_best_action(q_table[state])

        return action


class QLearner:
    def __init__(self, discount_rate, exploration_strategy, environment, min_alpha=0.1):
        self.discount_rate = discount_rate
        self.exploration_strategy = exploration_strategy
        self.environment = environment
        self.min_alpha = min_alpha

        Q_shape = (environment.state_space, len(environment.action_space))
        self.q = np.zeros(Q_shape)
        self.n = np.zeros(Q_shape)

    def choose_action(self, state):
        action = self.exploration_strategy(state, self.q)
        return action

    def update(self, s, a, r, s_prime):
        # update count
        self.n[s, a] += 1

        current_q = self.q[s, a]
        alpha = max(1 / self.n[s, a], self.min_alpha)
        self.q[s, a] = (1 - alpha) * current_q + alpha * (
            r + self.discount_rate * self.q[s_prime].max()
        )

    def create_policy_func(self):
        """create_policy_func 

        Takes the current Q table and creates a policy function"""

        def policy(state):
            return self.q[state].argmax()

        return policy

    @staticmethod
    def create_policy_func_from_q(q):

        def policy(state):
            return q[state].argmax()

        return policy

    def run_until_convergence(
        self, tolerance=0.001, convergence_tests=50, verbose=True
    ):
        converged = False
        terminal = True
        prev_Q = self.q.copy()

        iterations = 0
        episodes = 0

        tolerance_count = 0

        differences = []
        while not converged:
            if terminal:
                # Run episode
                s = self.environment.reset()
                assert isinstance(s, int), "state must be in integer format"
                terminal = False
                if iterations != 0:
                    # Don't test convergence here
                    mse = np.sqrt((self.q - prev_Q) ** 2).mean().mean()
                    if mse < tolerance:
                        if tolerance_count > convergence_tests:
                            converged = True
                        else:
                            tolerance_count += 1
                    else:
                        tolerance_count = 0

                    if verbose:
                        print(f"MSE on episode {episodes+1}: {mse}")
                    differences.append(mse)
                    episodes += 1
                    prev_Q = self.q.copy()
            else:
                action = self.choose_action(s)
                s_p, r, terminal = self.environment.step(action)
                self.update(s, action, r, s_p)
                s = s_p
                iterations += 1

        return iterations, episodes, differences

    def run_x_episodes(self, n_episodes=1000, verbose=True):
        terminal = True

        episodes = -1 # Will immediately be incremented to zero

        while episodes < n_episodes:
            if terminal:
                # Run episode
                s = self.environment.reset()
                assert isinstance(s, int), "state must be in integer format"
                terminal = False
                episodes += 1
            else:
                action = self.choose_action(s)
                s_p, r, terminal = self.environment.step(action)
                self.update(s, action, r, s_p)
                s = s_p
