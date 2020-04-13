import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Circle
from matplotlib.ticker import AutoLocator
from matplotlib.colors import LinearSegmentedColormap
import os
import numpy as np
import pickle
from collections import defaultdict
import FrozenLake
from matplotlib import colors
from FrozenLake import get_small_lake, get_large_lake
from experiments import create_small_hunting_environment, extract_episode
from hunting_utils import get_solver_stats_by_animal
import seaborn as sns
import PIL

# Vi/Pi Plots

# Frozen Lake Plots

input_dir = "output"
output_dir = "plots"

frozenlake_dir = os.path.join(input_dir, "frozenlake")
hunter_dir = os.path.join(input_dir, "hunterschoice")


small_filenames = [
    os.path.join(frozenlake_dir, f)
    for f in os.listdir(frozenlake_dir)
    if "smalllake" in f and "evaluation" not in f
]
large_filenames = [
    os.path.join(frozenlake_dir, f)
    for f in os.listdir(frozenlake_dir)
    if "largelake" in f and "evaluation" not in f
]
small_hunter_filenames = [
    os.path.join(hunter_dir, f)
    for f in os.listdir(hunter_dir)
    if "small" in f and "evaluation" not in f
]

def get_small_hunter_stats_policies():
    small_hunter_policies = defaultdict(list)
    small_hunter_statistics = defaultdict(list)
    for f in small_hunter_filenames:
        _, solver_type, discount = f.split("_")
        with open(f, "rb") as infile:
            solver = pickle.load(infile)

            small_hunter_statistics[solver_type].append(
                {"discount": discount, "iters": solver.iter, "time": solver.time}
            )
            if solver_type == "vi":
                small_hunter_policies[solver_type].append(
                    {"policy": solver.policy, "discount": discount, "values": solver.V}
                )
            else:
                small_hunter_policies[solver_type].append(
                    {"policy": solver.policy, "discount": discount}
                )

    return small_hunter_statistics, small_hunter_policies

def get_small_lake_stats_policies():
    small_statistics = defaultdict(list)
    small_policies = defaultdict(list)
    for f in small_filenames:
        _, solver_type, discount = f.split("_")
        with open(f, "rb") as infile:
            solver = pickle.load(infile)

            small_statistics[solver_type].append(
                {"discount": discount, "iters": solver.iter, "time": solver.time}
            )
            if solver_type == "vi":
                small_policies[solver_type].append(
                    {"policy": solver.policy, "discount": discount, "values": solver.V}
                )
            else:
                small_policies[solver_type].append(
                    {"policy": solver.policy, "discount": discount}
            )

    return small_statistics, small_policies


def get_large_lake_stats_policies():
    large_statistics = defaultdict(list)

    large_policies = defaultdict(list)

    for f in large_filenames:
        _, solver_type, discount = f.split("_")
        with open(f, "rb") as infile:
            solver = pickle.load(infile)

            large_statistics[solver_type].append(
                {"discount": discount, "iters": solver.iter, "time": solver.time}
            )
            if solver_type == "vi":
                large_policies[solver_type].append(
                    {"policy": solver.policy, "discount": discount, "values": solver.V}
                )
            else:
                large_policies[solver_type].append(
                    {"policy": solver.policy, "discount": discount}
                )

    return large_statistics, large_policies



def convert_map_to_integers(m):

    conversion_map = {"F": 0, "H": 1, "S": 2, "G": 3}
    size = len(m[0])
    intmap = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            value = m[i][j]
            intmap[i, j] = conversion_map[value]

    return intmap


def render_map(title, frozenlake_map):
    intmap = convert_map_to_integers(frozenlake_map)
    size = len(frozenlake_map)
    # create discrete colormap
    cmap = colors.ListedColormap(["white", "black", "yellow", "green"])
    bounds = [0, 1, 2, 3, 4]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots()
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
    outpath = os.path.join(output_dir, title + ".png")
    plt.savefig(outpath)
    plt.close()


def render_frozenlake_policy(title, frozenlake_map, policy, values=None):
    intmap = convert_map_to_integers(frozenlake_map)
    size = len(frozenlake_map)
    # create discrete colormap

    fig, ax = plt.subplots()
    if values == None:
        cmap = colors.ListedColormap(["white", "black", "yellow", "green"])
        bounds = [0, 1, 2, 3, 4]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        ax.imshow(intmap, cmap=cmap, norm=norm)

    else:
        arr_values = np.array(values).reshape((size, size))
        ax.imshow(arr_values, cmap="Blues")

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


# example_values = small_policies['vi'][1]['values']
# example_policy = small_policies['vi'][1]['policy']
# render_frozenlake_policy('Large Lake Optimal Policy', small_lake.m, example_policy, example_values)

# Plot # Of iterations / Discount Rate
#
def get_stats(stats_dict):
    discount_rates = np.array(sorted([float(x["discount"]) for x in stats_dict["vi"]]))

    vi_stats = sorted(stats_dict["vi"], key=lambda x: x["discount"])
    pi_stats = sorted(stats_dict["pi"], key=lambda x: x["discount"])

    vi_iters = [v["iters"] for v in vi_stats]
    vi_time = [v["time"] for v in vi_stats]

    pi_iters = [p["iters"] for p in pi_stats]
    pi_time = [p["time"] for p in pi_stats]

    return discount_rates, vi_iters, vi_time, pi_iters, pi_time


def plot_comparison(
    title,
    discount_rates,
    vi_vals,
    pi_vals,
    ylabel,
    logy=False,
    vi_err=None,
    pi_err=None,
):
    assert (vi_err is not None and pi_err is not None) or (
        vi_err is None and pi_err is None
    ), "Must provide either both errors or neither"
    fig = plt.figure()
    width = 0.45

    ax = fig.add_subplot(111)
    xs = np.arange(len(discount_rates))
    ax.bar(xs + width / 2, vi_vals, width, color="blue", label="Value Iteration")
    ax.bar(xs - width / 2, pi_vals, width, color="green", label="Policy Iteration")

    if vi_err is not None:
        ax.errorbar(xs + width / 2, vi_vals, vi_err, ecolor="black", fmt="none")
        ax.errorbar(xs - width / 2, pi_vals, pi_err, ecolor="black", fmt="none")

    ax.set_xticks(xs)
    ax.set_xticklabels(discount_rates)

    ax.set_ylabel(ylabel)
    ax.set_xlabel("Discount Rate")

    if logy:
        ax.set_yscale("log")

    ax.set_title(title)
    ax.legend()

    outpath = os.path.join(output_dir, title + ".png")
    plt.savefig(outpath)
    plt.close()


def plot_frozenlake_stats():
    (
        discount_rates,
        small_vi_iters,
        small_vi_time,
        small_pi_iters,
        small_pi_time,
    ) = get_stats(small_statistics)
    (
        discount_rates,
        large_vi_iters,
        large_vi_time,
        large_pi_iters,
        large_pi_time,
    ) = get_stats(large_statistics)

    plot_comparison(
        "Frozen Lake (Small) Time until Convergence",
        discount_rates,
        small_vi_time,
        small_pi_time,
        "Time Elapsed (s)",
    )
    plot_comparison(
        "Frozen Lake (Small) Iterations until Convergence",
        discount_rates,
        small_vi_iters,
        small_pi_iters,
        "Iterations",
        logy=True,
    )

    plot_comparison(
        "Frozen Lake (Large) Time until Convergence",
        discount_rates,
        large_vi_time,
        large_pi_time,
        "Time Elapsed (s)",
    )
    plot_comparison(
        "Frozen Lake (Large) Iterations until Convergence",
        discount_rates,
        large_vi_iters,
        large_pi_iters,
        "Iterations",
        logy=True,
    )


def plot_hunterschoice_stats():
    small_hunter_statistics, _ = get_small_hunter_stats_policies()
    (
        discount_rates,
        small_vi_iters,
        small_vi_time,
        small_pi_iters,
        small_pi_time,
    ) = get_stats(small_hunter_statistics)

    plot_comparison(
            "Hunter's Choice: Time until Convergence",
        discount_rates,
        small_vi_time,
        small_pi_time,
        "Time Elapsed (s)",
    )
    plot_comparison(
        "Hunter's Choice Iterations until Convergence",
        discount_rates,
        small_vi_iters,
        small_pi_iters,
        "Iterations",
        logy=True,
    )



def plot_frozenlake_performance(policy):
    pass


discount_rate_extraction = lambda x: float(x.split("_")[2])


def get_eval_stats(evals, size, solver_type):
    stats = [x for x in evals if size in x and solver_type in x]

    return sorted(stats, key=discount_rate_extraction)


def plot_success_rates(env="FrozenLake"):
    assert env in ["FrozenLake", "HuntersChoice"], "env not supported"
    if env == "FrozenLake":
        evals = [
            os.path.join(frozenlake_dir, x)
            for x in os.listdir(frozenlake_dir)
            if "evaluation" in x
        ]

    else:
        evals = [
            os.path.join(hunter_dir, x)
            for x in os.listdir(hunter_dir)
            if "evaluation" in x
        ]
    small_vi = get_eval_stats(evals, "small", "vi")
    large_vi = get_eval_stats(evals, "large", "vi")
    small_pi = get_eval_stats(evals, "small", "pi")
    large_pi = get_eval_stats(evals, "large", "pi")

    # Plot small vis/small pis
    fig = plt.figure()
    ax = fig.add_subplot(111)

    def load_success_rates(evaluation_files):
        success_rates = []
        for eval_file in evaluation_files:
            with open(eval_file, "rb") as f:
                vi_data = pickle.load(f)
                success_rates.append(vi_data["success_rate"])

        return success_rates

    def load_episode_lengths(evaluation_files):
        success_rates = []
        for eval_file in evaluation_files:
            with open(eval_file, "rb") as f:
                vi_data = pickle.load(f)
                success_rates.append(vi_data["episode_lengths"])

        return success_rates

    discount_rates = [discount_rate_extraction(d) for d in small_vi]

    if env == "FrozenLake":
        small_vi_vals = load_success_rates(small_vi)
        small_pi_vals = load_success_rates(small_pi)

        large_vi_vals = load_success_rates(large_vi)
        large_pi_vals = load_success_rates(large_pi)

        small_title = f"Success Rate of Agent Using Extracted Policy (Small Version)"
        big_title = f"Success Rate of Agent Using Extracted Policy (Large Version)"

        small_vi_err = None
        small_pi_err = None
        large_vi_err = None
        large_pi_err = None

        ylabel = 'Success Rate'
    else:
        small_vi_lengths = load_episode_lengths(small_vi)
        small_vi_vals = np.mean(small_vi_lengths, axis=1)
        small_vi_err = np.std(small_vi_lengths, axis=1)

        small_pi_lengths = load_episode_lengths(small_pi)
        small_pi_vals = np.mean(small_pi_lengths, axis=1)
        small_pi_err = np.std(small_pi_lengths, axis=1)

        small_title = f"Survival Time of Agent Using Extracted Policy"
        ylabel='Survival Time'

    plot_comparison(
        small_title,
        discount_rates,
        small_vi_vals,
        small_pi_vals,
        ylabel,
        vi_err=small_vi_err,
        pi_err=small_pi_err,
    )

    if env == "FrozenLake":
        plot_comparison(
            big_title,
            discount_rates,
            large_vi_vals,
            large_pi_vals,
            ylabel,
            vi_err=large_vi_err,
            pi_err=large_pi_err,
        )


def render_frozenlake_policies(title, discount_rate, size):
    small_lake = get_small_lake()
    large_lake = get_large_lake()
    if size == "small":
        vi_policy = [
            solver["policy"]
            for solver in small_policies["vi"]
            if float(solver["discount"]) == discount_rate
        ][0]
        vi_values = [
            solver["values"]
            for solver in small_policies["vi"]
            if float(solver["discount"]) == discount_rate
        ][0]
        pi_policy = [
            solver["policy"]
            for solver in small_policies["pi"]
            if float(solver["discount"]) == discount_rate
        ][0]


        render_frozenlake_policy(
            f"Value Iteration {title} (Small)",
            small_lake.m,
            vi_policy,
            values=vi_values,
        )
        render_frozenlake_policy(
            f"Policy Iteration {title} (Small)", small_lake.m, pi_policy
        )
    else:
        vi_policy = [
            solver["policy"]
            for solver in large_policies["vi"]
            if float(solver["discount"]) == discount_rate
        ][0]
        vi_values = [
            solver["values"]
            for solver in large_policies["vi"]
            if float(solver["discount"]) == discount_rate
        ][0]
        pi_policy = [
            solver["policy"]
            for solver in large_policies["pi"]
            if float(solver["discount"]) == discount_rate
        ][0]

        render_frozenlake_policy(
            f"Value Iteration {title} (Large)",
            large_lake.m,
            vi_policy,
            values=vi_values,
        )
        render_frozenlake_policy(
            f"Policy Iteration {title} (Large)", large_lake.m, pi_policy
        )

def render_hunting_policy(title, policy):
    """render_hunting_policy

    This was pure hell to make. Renders a visual representation of a hunting policy.

    Parameters
    ----------

    title : Title of figure
    policy :

    Returns
    -------
    """
    env = create_small_hunting_environment()
    p_stats, v_stats = get_solver_stats_by_animal(env, policy)

    fig, axs = plt.subplots(1, 5, sharey=True, figsize=(8, 10))

    cmap = LinearSegmentedColormap.from_list("Custom", ("goldenrod", "purple"), 2)

    buffalo = p_stats["buffalo"]
    ostrich = p_stats["ostrich"]
    lemur = p_stats["lemur"]
    rabbit = p_stats["rabbit"]
    bird = p_stats["bird"]

    cmap = colors.ListedColormap(["goldenrod", "red", "purple"])
    bounds = [0, 1, 2]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    for ax, intmap, name in zip(
        axs,
        [buffalo, ostrich, lemur, rabbit, bird],
        ["buffalo", "ostrich", "lemur", "rabbit", "bird"],
    ):
        values = None
        if values == None:
            ax.imshow(intmap, cmap=cmap, norm=norm)
        else:
            arr_values = np.array(values).reshape((size, size))
            ax.imshow(arr_values, cmap="Blues")

        # draw gridlines
        rows = 50
        cols = 5
        ax.grid(which="major", axis="both", linestyle="-", color="k", linewidth=2)
        ax.set_xticks(np.arange(-0.5, cols, 1))
        ax.set_yticks(np.arange(-0.5, rows, 1))

        ax.set_xticklabels(np.arange(0, cols + 1, 1))
        ax.set_yticklabels(np.arange(0, rows + 1, 1))

        ax.set_xlabel("Injury")

        ax.xaxis.set_tick_params(size=0)
        ax.yaxis.set_tick_params(size=0)

        for label in ax.xaxis.get_majorticklabels():
            label.set_horizontalalignment("left")

        for label in ax.yaxis.get_majorticklabels():
            label.set_verticalalignment("top")

        ax.set_title(name)

    axs[0].set_ylabel('Energy')
    legend_elements = [
        Patch(facecolor="goldenrod", edgecolor="goldenrod", label="HUNT"),
        Patch(facecolor="purple", edgecolor="purple", label="WAIT"),
    ]
    axs[2].legend(
        handles=legend_elements, loc="center left", bbox_to_anchor=(-0.06, 1.08)
    )
    plt.suptitle(title)

    outpath = os.path.join(output_dir, title + ".png")
    plt.savefig(outpath)
    plt.close()


def render_hunting_policies():
    _, small_hunter_policies = get_small_hunter_stats_policies()
    pi999_policy = [x['policy'] for x in small_hunter_policies['pi'] if x['discount'] == str(0.999)][0]
    vi999_policy = [x['policy'] for x in small_hunter_policies['vi'] if x['discount'] == str(0.999)][0]
    vi999_value = [x['values'] for x in small_hunter_policies['vi'] if x['discount'] == str(0.999)][0]

    pi1_policy = [x['policy'] for x in small_hunter_policies['pi'] if x['discount'] == str(0.1)][0]
    vi1_policy = [x['policy'] for x in small_hunter_policies['vi'] if x['discount'] == str(0.1)][0]
    vi1_value = [x['values'] for x in small_hunter_policies['vi'] if x['discount'] == str(0.1)][0]

    render_hunting_policy('Optimal Policy (Policy Iteration)', pi999_policy)
    render_hunting_policy('Optimal Policy (Value Iteration)', vi999_policy)

    render_hunting_policy('Worst Policy (Policy Iteration)', pi1_policy)
    render_hunting_policy('Worst Policy (Value Iteration)', vi1_policy)




