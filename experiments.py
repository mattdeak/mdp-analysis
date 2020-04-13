from HuntingMDP import HuntingMDP, Animal
from tqdm import tqdm
from Wrappers import HuntingMDPWrapper, RewardShapedFrozenLake
from utils import get_vipi_results, Agent, run_agent
from QLearning import (
    EpsilonGreedyDecay,
    EpsilonGreedyConstant,
    EpsilonGreedyStateBasedDecay,
    GreedyExploration,
    RandomExploration,
    EqualExploration,
    QLearner,
)
import FrozenLake
from FrozenLake import (
    FrozenLake,
    get_small_lake,
    get_large_lake,
    create_frozenlake_policy,
)
import pickle
import os
import numpy as np
from utils import listdir

output_dir = "output"
frozenlake_dir = os.path.join(output_dir, "frozenlake")
hunter_dir = os.path.join(output_dir, "hunterschoice")


def create_hunterschoice_policy(raw_policy):
    """create_hunterschoice_policy

    Parameters
    ----------

    policy : ValueIteration.policy or PolicyIteration.policy

    Returns
    Env-compatible policy
    -------
    """

    env = create_small_hunting_environment()

    def policy(state):
        s = env.state_to_idx[state]
        return raw_policy[s]

    return policy


def create_small_hunting_environment():
    # Base Environment
    buffalo = Animal("buffalo", 30, 0.2, 0.5)
    rabbit = Animal("rabbit", 9, 0.35, 0.0)
    lemur = Animal("lemur", 12, 0.4, 0.05)
    bird = Animal("bird", 7, 0.25, 0)
    ostrich = Animal("ostrich", 20, 0.4, 0.3)

    animal_names = ["buffalo", "rabbit", "lemur", "bird", "ostrich"]

    environment = HuntingMDP(
        [buffalo, rabbit, lemur, bird, ostrich],
        [0.05, 0.25, 0.1, 0.2, 0.08],
        injured_penalty=0.1,
    )

    return environment




def run_frozenlake_solvers():
    # Get frozen lake statistics
    small_lake = get_small_lake()
    large_lake = get_large_lake()

    small_lake_results = get_vipi_results(small_lake, max_iter=20000)
    large_lake_results = get_vipi_results(large_lake, max_iter=20000)

    for k, solver in small_lake_results.items():
        solver_type, discount, _ = k.split("_")

        with open(
            os.path.join(frozenlake_dir, f"smalllake_{solver_type}_{discount}"), "wb"
        ) as f:
            pickle.dump(solver, f)

    for k, solver in large_lake_results.items():
        solver_type, discount, _ = k.split("_")

        with open(
            os.path.join(frozenlake_dir, f"largelake_{solver_type}_{discount}"), "wb"
        ) as f:
            pickle.dump(solver, f)


def simulate_frozenlake_policy(policy_location, N=1000):
    print(f"Simulating Policy {policy_location}")
    with open(policy_location, "rb") as f:
        solver = pickle.load(f)

    if "small" in policy_location:
        lake = get_small_lake()
    else:
        lake = get_large_lake()

    policy = create_frozenlake_policy(solver.policy)
    agent = Agent(lake, policy)

    successes = 0
    episode_lengths = []
    for i in range(N):
        rewards = run_agent(agent)
        if rewards[-1] == 1:
            successes += 1
        episode_lengths.append(len(rewards))

    return successes / N, episode_lengths


def simulate_all_frozenlake_policies():
    policies = [
        os.path.join(frozenlake_dir, x)
        for x in listdir(frozenlake_dir)
        if "evaluation" not in x
    ]
    for policy in policies:
        success_rate, episode_lengths = simulate_frozenlake_policy(policy)
        result_dict = {"success_rate": success_rate, "episode_lengths": episode_lengths}
        out_dir = policy + "_evaluation"
        with open(out_dir, "wb") as f:
            pickle.dump(result_dict, f)


def run_hunterschoice_solvers():
    small_huntinggrounds = create_small_hunting_environment()

    small_hunter_results = get_vipi_results(small_huntinggrounds, max_iter=20000)

    for k, solver in small_hunter_results.items():
        solver_type, discount, _ = k.split("_")

        with open(
            os.path.join(hunter_dir, f"smallhunter_{solver_type}_{discount}"), "wb"
        ) as f:
            pickle.dump(solver, f)


def simulate_hunterschoice_policy(policy_location, N=1000):
    print(f"Simulating Policy {policy_location}")
    with open(policy_location, "rb") as f:
        solver = pickle.load(f)

    grounds = create_small_hunting_environment()
    policy = create_hunterschoice_policy(solver.policy)

    agent = Agent(grounds, policy)

    successes = 0
    episode_lengths = []
    for i in range(N):
        rewards = run_agent(agent)
        if rewards[-1] == 1:
            successes += 1
        episode_lengths.append(len(rewards))

    return successes / N, episode_lengths


def simulate_all_hunterschoice_policies():
    policies = [
        os.path.join(hunter_dir, x)
        for x in listdir(hunter_dir)
        if "evaluation" not in x
    ]
    for policy in policies:
        success_rate, episode_lengths = simulate_hunterschoice_policy(policy)
        result_dict = {"success_rate": success_rate, "episode_lengths": episode_lengths}
        out_dir = policy + "_evaluation"
        with open(out_dir, "wb") as f:
            pickle.dump(result_dict, f)


def collect_qlearner_data(
    env,
    exploration_strategy,
    save_dir,
    save_episode_rate=5000,
    max_episodes=300000,
    discount_rate=0.999,
):

    ql = QLearner(discount_rate, exploration_strategy, env)

    for i in range(max_episodes // save_episode_rate):
        print(f"Running Episodes {i*save_episode_rate + 1} - {(i+1)*save_episode_rate}")
        ql.run_x_episodes(n_episodes=save_episode_rate)

        save_path = os.path.join(save_dir, f"episode{(i+1)*save_episode_rate}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(ql.q, f)


def run_all_qlearners(
    discount=0.999,
    save_episode_rate=500,
    max_episodes=20000,
    envs=["hunter", "smalllake", "largelake", "shaped_smalllake", "shaped_largelake"],
):

    hunter_raw = create_small_hunting_environment()
    hunter = HuntingMDPWrapper(hunter_raw)
    small_lake = get_small_lake()
    large_lake = get_large_lake()

    shaped_smallake = RewardShapedFrozenLake(small_lake)
    shaped_largelake = RewardShapedFrozenLake(large_lake)

    randomexploration = RandomExploration()
    equalexploration = EqualExploration()
    greedyexploration = GreedyExploration()
    epsilongreedy = EpsilonGreedyConstant(0.1)
    epsilondecay = EpsilonGreedyDecay(decay_rate=0.00001)
    epsilonstatedecay = EpsilonGreedyStateBasedDecay()

    env_map = {
        "hunter": hunter,
        "smalllake": small_lake,
        "largeuake": shaped_largelake,
        "shaped_smalllake": shaped_smallake,
        "shaped_largelake": shaped_largelake,
    }

    for sub_dir in envs:
        env = env_map[sub_dir]
        root_dir = os.path.join(output_dir, "qlearning", sub_dir)
        for exploration_strategy, explore_dir in zip(
            [
                randomexploration,
                equalexploration,
                greedyexploration,
                epsilongreedy,
                epsilondecay,
                epsilonstatedecay,
            ],
            [
                "random",
                "equal",
                "greedy",
                "epsilongreedy",
                "epsilondecay",
                "epsilonstatedecay",
            ],
        ):
            print(f"Collecting Data For {sub_dir} with strategy {explore_dir}")

            save_dir = os.path.join(root_dir, explore_dir)
            collect_qlearner_data(
                env,
                exploration_strategy,
                save_dir,
                save_episode_rate=save_episode_rate,
                max_episodes=max_episodes,
                discount_rate=discount,
            )


def reset_qlearner_folders():
    for sub_dir in [
        "hunter",
        "smalllake",
        "largelake",
        "shaped_smalllake",
        "shaped_largelake",
    ]:
        whole_sub_dir = os.path.join(output_dir, 'qlearning',sub_dir)
        if not os.path.exists(whole_sub_dir):
            os.mkdir(whole_sub_dir)

        for explore_dir in [
            "random",
            "equal",
            "greedy",
            "epsilongreedy",
            "epsilondecay",
            "epsilonstatedecay",
        ]:

            clear_dir = os.path.join(output_dir, "qlearning", sub_dir, explore_dir)
            if not os.path.exists(clear_dir):
                os.mkdir(clear_dir)
                continue

            for filename in listdir(clear_dir):
                os.unlink(os.path.join(clear_dir, filename))


# ql.run_until_convergence(tolerance=0.001, convergence_tests=1000)
def extract_episode(filepath):
    episode_start = filepath.find("episode") + len("episode")
    pkl_start = filepath.find(".pkl")
    return int(filepath[episode_start:pkl_start])


def get_sorted_qlearner_filepaths(directory):
    files = listdir(directory)
    files = sorted(files, key=extract_episode)
    return files


def load_pickle(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)


def simulate_hunterschoice_policy_from_func(policy, N=1000):
    grounds = create_small_hunting_environment()
    env = HuntingMDPWrapper(grounds)
    agent = Agent(env, policy)

    successes = 0
    episode_lengths = []
    for i in range(N):
        rewards = run_agent(agent)
        if rewards[-1] == 1:
            successes += 1
        episode_lengths.append(len(rewards))

    return successes / N, episode_lengths


def simulate_lake_policy_from_func(policy, N=1000, size="small"):
    if size == "small":
        env = get_small_lake()
    else:
        env = get_large_lake()
    agent = Agent(env, policy)

    successes = 0
    episode_lengths = []
    for i in range(N):
        rewards = run_agent(agent)
        if rewards[-1] == 1:
            successes += 1
        episode_lengths.append(len(rewards))

    return successes / N, episode_lengths


def save_qlearner_stats(qlearner_dir, out_path, problem="smalllake", stopper=5000):
    learners = get_sorted_qlearner_filepaths(qlearner_dir)
    learners = [l for l in learners if extract_episode(l) % stopper == 0]

    stats = {}
    for learner in tqdm(learners):
        episode = extract_episode(learner)
        q = load_pickle(os.path.join(qlearner_dir, learner))
        policy = QLearner.create_policy_func_from_q(q)
        if problem == "smalllake" or problem == "shaped_smalllake":
            ql_stats = simulate_lake_policy_from_func(policy)
        elif problem == "largelake" or problem == "shaped_largelake":
            ql_stats = simulate_lake_policy_from_func(policy, size="large")
        else:
            ql_stats = simulate_hunterschoice_policy_from_func(policy)
        stats[episode] = ql_stats

    with open(out_path, "wb") as f:
        pickle.dump(stats, f)

    return stats  # For debugging


def save_all_qlearner_stats(
    envs=["hunter", "smalllake", "largelake", "shaped_smalllake", "shaped_largelake"],
    strategy_names=[
        "random",
        "equal",
        "greedy",
        "epsilongreedy",
        "epsilondecay",
        "epsilonstatedecay",
    ],
    stopper=5000,
):
    for env in envs:
        for st_name in strategy_names:
            print(f"Collecting QLearner stats for {env}, {st_name} strategy")
            outfile = os.path.join("output", "qlearning_sims", f"{env}_{st_name}.pkl")
            qlearner_dir = f"output/qlearning/{env}/{st_name}"
            stats = save_qlearner_stats(
                qlearner_dir, outfile, problem=env, stopper=stopper
            )


# save_all_qlearner_stats(envs=['shaped_smalllake','shaped_largelake'],

# Plot a policy comparison
# Plot a verr comparison
def run_convergence_experiment(env_name,N=20000, stopper=1):
    best_explorer_lookup_table = {
        "hunter": EqualExploration(),
        "smalllake": EpsilonGreedyConstant(0.1),
        "largelake": EpsilonGreedyConstant(0.1),
    }

    env_lookup = {
        "hunter": HuntingMDPWrapper(create_small_hunting_environment()),
        "smalllake": RewardShapedFrozenLake(get_small_lake()),
        "largelake": RewardShapedFrozenLake(get_large_lake()),
    }

    explorer = best_explorer_lookup_table[env_name]
    env = env_lookup[env_name]

    ql = QLearner(0.999, explorer, env)
    diffs = {}
    prev_Q = ql.q.copy()
    for i in tqdm(range(N//stopper)):
        ql.run_x_episodes(stopper)
        diff = abs(ql.q - prev_Q).max()
        prev_Q = ql.q.copy()
        diffs[i] = diff

    outfile = f'{env_name}_convergence.pkl'
    outpath = os.path.join('output','qlearning_convergence', outfile)
    with open(outpath, 'wb') as f:
        pickle.dump(diffs, f)

    return diffs


def run_all_convergence_experiments():
    run_convergence_experiment('hunter')
    run_convergence_experiment('smalllake')
    run_convergence_experiment('largelake')


def run_best_qlearner_vs_optimal_experiment(env):
    best_q_lookup = {
            'hunter':'output/qlearning/hunter/equal',
            'smalllake':'output/qlearning/shaped_smalllake/epsilonstatedecay',
            'largelake':'output/qlearning/shaped_largelake/epsilonstatedecay'}
    env_lookup = {
            'smalllake': get_small_lake(),
            'largelake': get_large_lake()}

    best_vi_lookup = {
            'hunter':'output/hunterschoice/smallhunter_vi_0.999',
            'smalllake': 'output/frozenlake/smalllake_vi_0.999',
            'largelake': 'output/frozenlake/largelake_vi_0.999'}

    best_vi = best_vi_lookup[env]
    with open(best_vi, 'rb') as f:
        solver = pickle.load(f)

    indir = best_q_lookup[env]
    qlearners = get_sorted_qlearner_filepaths(indir)
    # Use last qlearner
    best = qlearners[-1]
    with open(os.path.join(indir, best), 'rb') as f:
        qt = pickle.load(f)

    def optimal_policy(state):
        return solver.policy[state]

    def learned_policy(state):
        return qt.argmax(axis=1)[state]


    if env == 'smalllake':
        optimal_results = simulate_lake_policy_from_func(optimal_policy, size='small',N=10000)
        learned_results = simulate_lake_policy_from_func(learned_policy, size='small',N=10000)
    elif env == 'largelake':
        optimal_results = simulate_lake_policy_from_func(optimal_policy, size='large',N=10000)
        learned_results = simulate_lake_policy_from_func(learned_policy, size='large',N=10000)
    elif env == 'hunter':
        optimal_results = simulate_hunterschoice_policy_from_func(optimal_policy, N=10000)
        learned_results = simulate_hunterschoice_policy_from_func(learned_policy, N=10000)

    outdir = os.path.join('output','final_evaluation')

    with open(os.path.join(outdir, f'{env}_optimal.pkl'), 'wb') as f:
        pickle.dump(optimal_results, f)

    with open(os.path.join(outdir, f'{env}_learned.pkl'), 'wb') as f:
        pickle.dump(learned_results, f)

def run_all_final_evaluations():
    for env in ['hunter','smalllake','largelake']:
        print(f"Running {env} final evaluation")
        run_best_qlearner_vs_optimal_experiment(env)
