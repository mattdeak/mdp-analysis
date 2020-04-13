import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib import colors
import FrozenLake
from scipy.stats import kendalltau
from experiments import create_small_hunting_environment, get_sorted_qlearner_filepaths, extract_episode, get_small_lake, get_large_lake
from plotting import render_hunting_policy
from hunting_utils import get_solver_stats_by_animal

output_dir = 'plots'

def get_stats(infile):
    with open(infile, 'rb') as f:
        stats = pickle.load(f)

    return stats


def extract_experiment_details(filename):
    file_without_path = filename.split('/')[-1]
    

# plot episode lengths

def collect_episode_lengths(stats):
    means = []
    medians = []
    stds = []
    for _, episode_lengths in stats.values():
        means.append(np.mean(episode_lengths))
        medians.append(np.median(episode_lengths))
        stds.append(np.std(episode_lengths))
    return np.array(means), np.array(medians), np.array(stds)

def collect_success_rates(stats):
    success_rates = []
    for success_rate, _ in stats.values():
        success_rates.append(success_rate)

    return success_rates


def plot_qlearner_evals(title, env='hunter', show=False):

    all_stats = {}

    for strategy in ['random','equal','greedy','epsilongreedy','epsilondecay','epsilonstatedecay']:
        infile = os.path.join('output','qlearning_sims',f'{env}_{strategy}.pkl')
        stats = get_stats(infile)
        all_stats[strategy] = stats


        

    fig, ax = plt.subplots(1, 1)


    episodes = np.array(list(all_stats[list(all_stats.keys())[0]].keys()))

    for strategy in all_stats.keys():
        episodes = np.array(list(all_stats[strategy].keys()))

        if env == 'hunter':
            y= 'Survival Time'
            means, medians, stds = collect_episode_lengths(all_stats[strategy])
            ax.plot(episodes, means, label=strategy)
            ax.fill_between(episodes, means - stds, means + stds, alpha=0.2)
        else:
            success_rates = collect_success_rates(all_stats[strategy])
            y = 'Success Rate'
            ax.plot(episodes, success_rates, label=strategy)

    ax.set_xlim(min(episodes), max(episodes))
    ax.set_ylim(bottom=0)

    ax.legend()

    ax.set_ylabel(y)
    ax.set_xlabel('Episodes')
    ax.set_title(title)

    outpath = os.path.join(output_dir, title + '.png')
    plt.savefig(outpath)
    if show:
        plt.show()

def plot_qlearner_maxqs(title, env='hunter', show=False):
    q_tables = {}

    for strategy in ['random','equal','greedy','epsilongreedy','epsilondecay','epsilonstatedecay']:
        infile = os.path.join('output','qlearning_sims',f'{env}_{strategy}.pkl')
        stats = get_stats(infile)
        all_stats[strategy] = stats

    fig, ax = plt.subplots(1, 1)


    episodes = np.array(list(all_stats[list(all_stats.keys())[0]].keys()))

    for strategy in all_stats.keys():
        episodes = np.array(list(all_stats[strategy].keys()))

        if env == 'hunter':
            y= 'Survival Time'
            means, medians, stds = collect_episode_lengths(all_stats[strategy])
            ax.plot(episodes, means, label=strategy)
            ax.fill_between(episodes, means - stds, means + stds, alpha=0.2)
        else:
            success_rates = collect_success_rates(all_stats[strategy])
            y = 'Success Rate'
            ax.plot(episodes, success_rates, label=strategy)

    ax.set_xlim(min(episodes), max(episodes))
    ax.set_ylim(bottom=0)

    ax.legend()

    ax.set_ylabel(y)
    ax.set_xlabel('Episodes')
    ax.set_title(title)

    outpath = os.path.join(output_dir, title + '.png')
    plt.savefig(outpath)
    if show:
        plt.show()



def plot_maxq_values(title, env):
    all_stats = {}
    for strategy in ['random','equal','greedy','epsilongreedy','epsilondecay','epsilonstatedecay']:
        indir = os.path.join('output','qlearning',f'{env}',f'{strategy}')
        qlearners = get_sorted_qlearner_filepaths(indir)
        maxqs = {}
        for learner in qlearners:
            episode = extract_episode(learner)
            path = os.path.join(indir, learner)
            with open(path, 'rb') as f:
                qt = pickle.load(f)

            maxq = qt.max().max()
            maxqs[episode] = maxq
        all_stats[strategy]=maxqs

    episodes = np.array(list(all_stats[list(all_stats.keys())[0]].keys()))

    fig, ax = plt.subplots(1, 1)
    for strategy, results in all_stats.items():
        ax.plot(episodes, list(results.values()), label=strategy)

    ax.legend()
    ax.set_xlim(episodes[0], episodes[-1])
    ax.set_ylim(bottom=0)
    ax.grid()
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Max Q-Value')

    ax.set_title(title)
    outpath = os.path.join(output_dir, title + ".png")
    plt.savefig(outpath)

def plot_all_maxqs():
    title_template = 'Max Q-Value over Time: {}'

    envs = ['hunter','smalllake','largelake','shaped_smalllake','shaped_largelake']
    titles = ["Hunter's Choice", "Small Lake", "Large Lake","Reward-Shaped Small Lake", "Reward-Shaped Large Lake"]
    for env, title_env in zip(envs, titles):
        title = title_template.format(title_env)
        plot_maxq_values(title, env)


def get_perrs(env):
    best_vi_lookup = {
            'hunter':'output/hunterschoice/smallhunter_vi_0.999',
            'smalllake': 'output/frozenlake/smalllake_vi_0.999',
            'largelake': 'output/frozenlake/largelake_vi_0.999'}

    best_q_lookup = {
            'hunter':'output/qlearning/hunter/equal',
            'smalllake':'output/qlearning/shaped_smalllake/epsilonstatedecay',
            'largelake':'output/qlearning/shaped_largelake/epsilonstatedecay'}

    best_vi = best_vi_lookup[env]
    with open(best_vi, 'rb') as f:
        solver = pickle.load(f)


    true_v = np.array(solver.V)


    true_policy = np.array(solver.policy)
    indir = best_q_lookup[env]
    qlearners = get_sorted_qlearner_filepaths(indir)
    perrs = {}
    for learner in qlearners:
        episode = extract_episode(learner)
        path = os.path.join(indir, learner)
        with open(path, 'rb') as f:
                qt = pickle.load(f)

                qv = qt.max(axis=1)

                qp = qt.argmax(axis=1)

                perr = np.sum(qp == true_policy) / qp.size

                perrs[episode] = perr

    return perrs

    ## plot value comparison ## plot policy comparison
    ##

def plot_perrs(title, env):
    hunter_perrs = get_perrs(env)
    

    episodes = [e for e in hunter_perrs]
    perrs = [v for v in hunter_perrs.values()]

    fig, ax = plt.subplots(1, 1)

    ax.set_ylim((0, 1))
    ax.set_xlim((episodes[0], episodes[-1]))
    ax.plot(episodes, perrs)
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Policy Score')
    ax.grid()

    ax.set_title(title)
    outpath = os.path.join(output_dir, title + '.png')
    plt.savefig(outpath)

def plot_all_perrs():
    plot_perrs("Policy Scores: Hunter's Choice", 'hunter')
    plot_perrs("Policy Scores: Small Lake", 'smalllake')
    plot_perrs("Policy Scores: Large Lake", 'largelake')


def convert_map_to_integers(m):

    conversion_map = {"F": 0, "H": 1, "S": 2, "G": 3}
    size = len(m[0])
    intmap = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            value = m[i][j]
            intmap[i, j] = conversion_map[value]

    return intmap
def render_frozenlake_policy(title, frozenlake_map, policy, reference=None):
    intmap = convert_map_to_integers(frozenlake_map)

    # If reference exists, turn policy differences red
    if reference is not None:

        policy_map = policy.reshape(intmap.shape)
        reference_map = reference.reshape(intmap.shape)
        intmap[policy_map != reference_map] = 4



    size = len(frozenlake_map)
    # create discrete colormap

    fig, ax = plt.subplots()
    cmap = colors.ListedColormap(["white", "black", "yellow", "green", "red"])
    bounds = [0, 1, 2, 3, 4, 5]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    ax.imshow(intmap, cmap=cmap, norm=norm)


    # draw gridlines
    ax.grid(which="major", axis="both", linestyle="-", color="k", linewidth=2)
    ax.set_xticks(np.arange(-0.5, size, 1))
    ax.set_yticks(np.arange(-0.5, size, 1))

    ax.xaxis.set_tick_params(size=0)
    ax.yaxis.set_tick_params(size=0)

    plt.title(title)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)

    for i in range(size):
        for j in range(size):
            if frozenlake_map[i][j] not in ["F", "S"]:
                continue
            arrow_direction = policy[i * size + j]
            if arrow_direction == FrozenLake.LEFT:
                # Left
                plt.arrow(
                    j + 0.25, i, -0.5, 0, length_includes_head=True, head_width=0.1
                )
            elif arrow_direction == FrozenLake.RIGHT:
                plt.arrow(
                    j - 0.25, i, 0.5, 0, length_includes_head=True, head_width=0.1
                )

            elif arrow_direction == FrozenLake.DOWN:
                plt.arrow(
                    j, i - 0.25, 0, 0.5, length_includes_head=True, head_width=0.1
                )
            elif arrow_direction == FrozenLake.UP:
                plt.arrow(
                    j, i + 0.25, 0, -0.5, length_includes_head=True, head_width=0.1
                )
            else:
                raise ValueError("Unexpected value in policy")

    outfile = os.path.join(output_dir, title + ".png")
    plt.savefig(outfile)
    plt.close()

def render_best_qlearner_policy(env):
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

    if env == 'smalllake':
        this_env = env_lookup[env]
        render_frozenlake_policy('Small Lake Q-Learned Policy', this_env.m, qt.argmax(axis=1), reference=np.array(solver.policy))
    elif env == 'largelake':
        this_env = env_lookup[env]
        render_frozenlake_policy('Large Lake Q-Learned Policy', this_env.m, qt.argmax(axis=1), reference=np.array(solver.policy))
    else:
        render_hunting_policy("Hunter's Choice Q-Learned Policy", qt.argmax(axis=1))


def plot_convergence_behaviour(env):
    diffs_lookup = {
            'hunter':'output/qlearning_convergence/hunter_convergence.pkl',
            'smalllake':'output/qlearning_convergence/smalllake_convergence.pkl',
            'largelake':'output/qlearning_convergence/largelake_convergence.pkl'}

    title_lookup = {
            'hunter': "Hunter's Choice Q-Learning Convergence",
            'smalllake': "Small Lake Q-Learning Convergence",
            'largelake': "Large Lake Q-Learning Convergence"}

    title = title_lookup[env]
    stat_path = diffs_lookup[env]
    
    with open(stat_path, 'rb') as f:
        stats = pickle.load(f)

    fig, ax = plt.subplots(1, 1)

    ax.set_ylabel('Max Q-Diff')
    ax.set_xlabel('Episodes')

    episodes = list(stats.keys())
    ax.set_xlim((episodes[0], episodes[-1]))
    vals = list(stats.values())
    ax.set_ylim((np.min(vals), np.max(vals)))

    smoothed = pd.Series(vals).rolling(50).mean()
    ax.plot(episodes, vals, label='Q-Diff')
    ax.plot(episodes, smoothed, label='Q-Diff Rolling Mean (50)')
    ax.set_title(title)
    ax.legend()
    ax.grid()

    outpath = os.path.join(output_dir, title + '.png')

    plt.savefig(outpath)
    plt.close()


def plot_all_convergences():
    plot_convergence_behaviour('hunter')
    plot_convergence_behaviour('smalllake')
    plot_convergence_behaviour('largelake')


# env = 'hunter'
# best_q_lookup = {
#         'hunter':'output/qlearning/hunter/equal',
#         'smalllake':'output/qlearning/shaped_smalllake/epsilonstatedecay',
#         'largelake':'output/qlearning/shaped_largelake/epsilonstatedecay'}
# env_lookup = {
#         'smalllake': get_small_lake(),
#         'largelake': get_large_lake()}

# best_vi_lookup = {
#         'hunter':'output/hunterschoice/smallhunter_vi_0.999',
#         'smalllake': 'output/frozenlake/smalllake_vi_0.999',
#         'largelake': 'output/frozenlake/largelake_vi_0.999'}

def print_final_eval_results(env):
    optimal_stat_lookup = {
            'hunter':'output/final_evaluation/hunter_optimal.pkl',
            'smalllake':'output/final_evaluation/smalllake_optimal.pkl',
            'largelake':'output/final_evaluation/largelake_optimal.pkl'}
    learned_stat_lookup = {
            'hunter':'output/final_evaluation/hunter_learned.pkl',
            'smalllake':'output/final_evaluation/smalllake_learned.pkl',
            'largelake':'output/final_evaluation/largelake_learned.pkl'}

    optimal_filepath = optimal_stat_lookup[env]
    learned_filepath = learned_stat_lookup[env]

    with open(optimal_filepath, 'rb') as f:
        optimal_stats = pickle.load(f)
        
    with open(learned_filepath, 'rb') as f:
        learned_stats = pickle.load(f)

    if env == 'hunter':
        optimal_svtime = np.mean(optimal_stats[1])
        learned_svtime = np.mean(learned_stats[1])

        optimal_svtime_std = np.std(optimal_stats[1])
        learned_svtime_std = np.std(learned_stats[1])
        print('Optimal Results')
        print(f'Mean Survival Time: {optimal_svtime}')
        print(f'STD Survival Time: {optimal_svtime_std}')

        print('------------')
        print('Learned Results')
        print(f'Mean Survival Time: {learned_svtime}')
        print(f'STD Survival Time: {learned_svtime_std}')
    else:
        optimal_srate = optimal_stats[0]
        learned_srate = learned_stats[0]
        
        print('Optimal Results')
        print(f"Success Rate: {optimal_srate}")

        print('------------')
        print('Learned Results')
        print(f"Success Rate: {learned_srate}")
