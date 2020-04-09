from HuntingMDP import HuntingMDP, Hunter, Animal
from Wrappers import HuntingMDPWrapper, FrozenLakeWrapper
from utils import get_vipi_results
from QLearning import EpsilonGreedyDecay, EpsilonGreedyConstant, EpsilonGreedyStateBasedDecay, QLearner

def create_small_hunting_environment():
    # Base Environment
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

    return environment

def create_large_hunting_environment():
    # Base Environment
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
        max_energy=150
    )

    return environment



# huntingmdp = create_base_environment()
# huntingwrapper = HuntingMDPWrapper(huntingmdp)

# exploration_strategy = EpsilonGreedyStateBasedDecay()

# DISCOUNT_RATE = 0.99
# ql = QLearner(DISCOUNT_RATE, exploration_strategy, huntingwrapper)

# ql.run_until_convergence(tolerance=0.001, convergence_tests=1000)
