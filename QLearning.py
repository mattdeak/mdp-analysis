import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict

class ExplorationStrategy(ABC):

    @abstractmethod
    def choose_action(self, state, q_table):
        """Choose an action for the state"""

    def __call__(self, state, q_table):
        # So you can use the class as a function
        return self.choose_action(state, q_table)

class EpsilonGreedyConstant(ExplorationStrategy):

    def __init__(self, epsilon):
        self.epsilon = epsilon

    def choose_action(self, state, q_table):
        rnd = np.random.random()
        if rnd < self.epsilon:
            action = np.random.randint(q_table.shape[1])
        else:
            action = q_table[state].argmax()
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
            action = q_table[state].argmax()

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
        epsilon = 1/self.state_counter[state]
        
        rnd = np.random.random()
        if rnd < epsilon:
            action = np.random.randint(q_table.shape[1])
        else:
            action = q_table[state].argmax()

        return action
    
class QLearner:
    def __init__(self, discount_rate, exploration_strategy, environment):
        self.discount_rate = discount_rate
        self.exploration_strategy = exploration_strategy
        self.environment = environment

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
        alpha = 1 / self.n[s, a]
        self.q[s, a] = (1 - alpha) * current_q + alpha * (
            r + self.discount_rate * self.q[s_prime].max()
        )

    def create_policy_func(self):
        """create_policy_func 

        Takes the current Q table and creates a policy function"""

        def policy(state):
            return self.q[state].argmax()

        return policy

    def run_until_convergence(self, tolerance=0.001, convergence_tests=50, verbose=True):
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
                        print(f'MSE on episode {episodes+1}: {mse}')
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
