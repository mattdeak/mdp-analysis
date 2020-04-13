import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mdptoolbox.mdp import ValueIteration
from mdptoolbox.mdp import PolicyIteration
from mdptoolbox.mdp import QLearning
from HuntingMDP import HuntingMDP, Animal
from utils import run_solver, get_vipi_results
from Wrappers import HuntingMDPWrapper
import pickle 
import os

output_dir = 'output'


def convert_solver_policy(environment, solver, *args, **kwargs):
    assert solver.policy is not None, "Solver must have run before converting policy"

    state_dict = environment.state_to_idx

    def policy(state):
        state_idx = state_dict[state]
        return solver.policy[state_idx]

    return policy


def run_hunter(hunter, verbose=False):
    t = False
    hunter.restart()
    iterations = 0
    energy_record = []
    injury_record = []
    choice_record = []
    while not t:
        obs = hunter.observation
        action = hunter.peek_action()

        if obs[0] == -1:
            animal = "None"
        else:
            animal = animal_names[obs[0]]

        if action == 0:
            choice = "HUNT"
        else:
            choice = "WAIT"

        energy = obs[1]
        injury = obs[2]
        if verbose:
            print(
                f"Animal Present: {animal}, Current Energy: {energy}, Current Injury Level: {injury}"
            )
            print(f"Choice: {choice}")

        choice_record.append((obs, action))
        s, r, t = hunter.act()
        energy_record.append(s[1])
        injury_record.append(s[2])
        iterations += 1

    return energy_record, injury_record, choice_record, iterations




def get_solver_stats_by_animal(environment, policy_fn, value_fn=None):

    max_energy = environment.max_energy
    max_injury = environment.recovery_time

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")

    animal_states = {}
    animal_names = [a.name for a in environment.animals]
    for i, animal_name in enumerate(animal_names):
        animal_states[animal_name] = [
            (i, energy, injury)
            for energy in range(max_energy + 1)
            for injury in range(max_injury + 1)
        ]


    animal_values = {}
    animal_policies = {}
    for animal_ix, (animal, states) in enumerate(animal_states.items()):
        values = np.zeros((max_energy + 1, max_injury + 1))
        policy = np.zeros((max_energy + 1, max_injury + 1))
        for i in range(max_energy + 1):
            for j in range(max_injury + 1):
                state = (animal_ix, i, j)
                state_idx = environment.state_to_idx[state]
                if value_fn is not None:
                    values[i, j] = value_fn[state_idx]
                policy[i, j] = policy_fn[state_idx]

        if value_fn is not None:
            animal_values[animal] = values
        animal_policies[animal] = policy

    if value_fn is not None:
        return animal_policies, animal_values
    else:
        return animal_policies, None

if __name__ == "__main__":
    print("Initializing Environment")
    buffalo = Animal(30, 0.2, 0.5)
    rabbit = Animal(9, 0.35, 0.0)
    lemur = Animal(12, 0.4, 0.05)
    bird = Animal(7, 0.25, 0)
    ostrich = Animal(20, 0.4, 0.3)


    animal_names = ["buffalo", "rabbit", "lemur", "bird", "ostrich"]

    environment = HuntingMDP(
        [buffalo, rabbit, lemur, bird, ostrich],
        [0.05, 0.25, 0.1, 0.2, 0.08],
        injured_penalty=0.1,
    )
    ql_wrapper = HuntingMDPWrapper(environment)


    discount_rates = [0.1, 0.5, 0.9, 0.99, 0.999]

    results = get_vipi_results(environment, max_iter=20000)
    hunter_results = {}
    for k, solver in results.items():
        print(f"Running Hunter using policy from solver {k}")
        policy = convert_solver_policy(environment, solver)
        hunter = Hunter(environment, policy)

        hunter_results[k] = simulate(hunter)

    with open(os.path.join(output_dir, "hunting_solvers.pkl"), 'wb') as f:
        pickle.dump(results, f)

    with open(os.path.join(output_dir, "hunter_simulations.pkl"), 'wb') as f:
        pickle.dump(hunter_results, f)
