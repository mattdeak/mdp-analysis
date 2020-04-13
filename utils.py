from mdptoolbox.mdp import ValueIteration, PolicyIteration


class Agent:
    def __init__(self, environment, policy):
        """__init__

        Parameters
        ----------

        environment : HunterMDP
        policy : Function that takes a state and returns an action

        Returns
        -------
        """
        self.observation = environment.reset()
        self.policy = policy
        self.environment = environment

    def act(self):
        action = self.policy(self.observation)
        self.observation, reward, terminal = self.environment.step(action)
        return self.observation, reward, terminal

    def peek_action(self):
        """peek_action

        Return the action given the current state, but don't perform it"""
        action = self.policy(self.observation)
        return action

    def restart(self):
        self.observation = self.environment.reset()


def run_agent(agent):
    agent.restart()
    reward_list = []

    t = False
    while not t:
        s, r, t = agent.act()
        reward_list.append(r)
    return reward_list


def get_vipi_results(
    environment, discount_rates=[0.1, 0.5, 0.9, 0.99, 0.999], **kwargs
):
    results = {}
    for discount in discount_rates:
        print(f"Running VI and PI with discount rate: {discount}")

        solvervi = run_solver(environment, ValueIteration, discount, **kwargs)

        solverpi = run_solver(
            environment, PolicyIteration, discount, eval_type=1, **kwargs
        )

        results[f"vi_{discount}_solver"] = solvervi
        results[f"pi_{discount}_solver"] = solverpi

    return results


def run_solver(environment, solver_func, *solver_args, **solver_kwargs):
    T, R = environment.build_TR_matrices()
    solver = solver_func(T, R, *solver_args, **solver_kwargs)

    solver.run()

    return solver
