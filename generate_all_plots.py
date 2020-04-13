from plotting import *
from plotting_utils import *


def generate_all():

    plot_frozenlake_stats()
    plot_hunterschoice_stats()

    render_frozenlake_policies('Frozen Lake Optimal Policy (small)',0.999,'small')
    render_frozenlake_policies('Frozen Lake Optimal Policy',0.999,'large')
    render_frozenlake_policies('Frozen Lake Worst Policy',0.1,'small')
    render_frozenlake_policies('Frozen Lake Worst Policy',0.1,'small')
    render_hunting_policies()

if __name__ == "__main__":
    generate_all()
