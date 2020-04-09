import matplotlib.pyplot as plt
import os
import numpy as np
import pickle
from collections import defaultdict
import FrozenLake
from matplotlib import colors
from FrozenLake import get_small_lake, get_large_lake
import PIL

# Vi/Pi Plots

# Frozen Lake Plots

input_dir = "output"
output_dir = "plots"


# Load Frozen Lake Statistics

small_statistics = defaultdict(list)
large_statistics = defaultdict(list)

small_policies = defaultdict(list)
large_policies = defaultdict(list)


small_filenames = [
    os.path.join(input_dir, f) for f in os.listdir(input_dir) if "smalllake" in f
]
large_filenames = [
    os.path.join(input_dir, f) for f in os.listdir(input_dir) if "largelake" in f
]

for f in small_filenames:
    _, solver_type, discount = f.split("_")
    with open(f, "rb") as infile:
        solver = pickle.load(infile)

        small_statistics[solver_type].append(
            {"discount": discount, "iters": solver.iter, "time": solver.time}
        )
        small_policies[solver_type].append(
            {"policy": solver.policy, "discount": discount}
        )


for f in large_filenames:
    _, solver_type, discount = f.split("_")
    with open(f, "rb") as infile:
        solver = pickle.load(infile)

        large_statistics[solver_type].append(
            {"discount": discount, "iters": solver.iter, "time": solver.time}
        )
        large_policies[solver_type].append(
            {"policy": solver.policy, "discount": discount}
        )


# Draw Policy
small_lake = get_small_lake()
large_lake = get_large_lake()

test_policy = small_policies["pi"][0]["policy"]


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
    plt.show()

def render_policy(title, frozenlake_map, policy):
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

    
    for i in range(size):
        for j in range(size):
            if frozenlake_map[i][j] not in ['F', 'S']:
                continue
            arrow_direction = policy[i*size + j]
            if arrow_direction == FrozenLake.LEFT:
                # Left
                plt.arrow(j+.25, i, -0.5, 0, length_includes_head=True, head_width=0.1)
            elif arrow_direction == FrozenLake.RIGHT:
                plt.arrow(j-0.25, i, 0.5, 0, length_includes_head=True, head_width=0.1)

            elif arrow_direction == FrozenLake.DOWN:
                plt.arrow(j, i-0.25, 0, 0.5, length_includes_head=True, head_width=0.1)
            elif arrow_direction == FrozenLake.UP:
                plt.arrow(j, i+0.25, 0, -0.5, length_includes_head=True, head_width=0.1)
            else:
                raise ValueError('Unexpected value in policy')

    plt.show()


example_policy = small_policies['pi'][0]['policy']
render_policy('Small Lake', small_lake.m, example_policy)
