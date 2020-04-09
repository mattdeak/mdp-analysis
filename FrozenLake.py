import gym
import numpy as np
from gym.envs.toy_text.frozen_lake import generate_random_map


m = generate_random_map(8)

m_small = [
    "SHFHFHFF",
    "FFHFFFFF",
    "FFFFFFFF",
    "FFHFFFFF",
    "FFHHHFHF",
    "FFFFFFHF",
    "HHFFFFHF",
    "FHFHFFFG",
]

m_large = [
    "SFFFHFHFFFFFFFFHHFHFFFHFFHHFFFHF",
    "FFFFHFFFHFFFFFFFFFFFFHFFHFHHFFHF",
    "FHFHFFHFFFFHFFFFFFFFHFFFHFFFFFFF",
    "FHHFFFFFFFFFFFFFHFFFHFFFFFFFFHFF",
    "FFFFHFFHFFFFFFFHFFFFFFFFFHFFFFFF",
    "FFHFFFFFFFFFHHFFFHFFHHFFFFHFFHFF",
    "FFFFHHFFFHFFHHFFHHFFFFHFHFFFHFFF",
    "FFHFFHFFFFHFFFHHFFHFHFFFFFFFFFFF",
    "FFFFFFFFFFFFFFFFFFFFFFHFHFHFFFFH",
    "FFFFFFFFFFFFHFFHFFHFHFFFFHFFFFHF",
    "HFFFHFFFFHFFFFFHFFFFFFFFFFFFFFFF",
    "FHFFFFFHHFFHFFFFFHFHFFHHFFFHFFFF",
    "HFFFFFHFHHFFHFFFFFFFHFFFFFFFFFHF",
    "FHFFFFFFFFFFFFFHFHHHFFFFFFHHHFFH",
    "FHFFFFFFHFFFFHHFFFFFFFFFFFFFHFFF",
    "FFFFFFHFFHFFFFFFFFFFFFFFHFFFFFFF",
    "FFFFFFFFFFFHFFFFHFHFFFFFFFFFFFFF",
    "FFFFFFFFHFFHFHHFFFFFFHFFHFFFFFFF",
    "HFHFFFHFHHFFFFFFFFHHFFFFFFFFFFHF",
    "FFFFFFHHFFFFFFFFFFFFHHFFFFFHFFHF",
    "HFFHFFFFFFFHFFFFFFFFHFFFFFFHFFFF",
    "FFFHFFFFFHFHFFFFFFFHFFFFHFHFHFFF",
    "FFHHFHFFHHHFHFFFFFFHFFFHFFHFFFFF",
    "FHHFFFFHFFFHFFHFHFFFFFFFFFHHFHFH",
    "FFFFFHFFFFFHFFFFFHHFHFFFFHFHFFFF",
    "FFFHFHHHFHFFFFFFFFFFFFFFFFFFHFFF",
    "FFHFFHHFFFFFFFFFHFFHFHFFHHHHFFHF",
    "HFFFFFFFHFFHFFFHHFFFFFHFFHFFFFFF",
    "FFHFFFFHFFHFFFHFFFHFFFHFHFFFFFHF",
    "HFHHFFHHHFFFFFFFFHFFFHFFHFHHFFFF",
    "FFHFFFHFHHHFFFFFFFFFFHHFFFFFHFFH",
    "FFHFHFFHFFFFFFFFFFFFFHFHFFFFFFFG",
]


LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

class FrozenLake:

    def __init__(self, m):
        self.m = m
        self.pos = 0

    def step(self, action):
        raise NotImplementedError

    def generate_TR_matrices(self):
        # Should generate the appropriate transition and reward matrices for any given map
        m = self.m # alias
        size = len(m)
        T = np.zeros((4, size * size, size * size))
        R = np.zeros(size * size)

        for i in range(size):
            for j in range(size):
                current_state = m[i][j]

                this_state_ix = i * size + j

                left_state_ix = i * size + max(j - 1, 0)
                right_state_ix = i * size + min(j + 1, size - 1)
                up_state_ix = max(i - 1, 0) * size + j
                down_state_ix = min(i + 1, size - 1) * size + j
                if current_state == "F" or current_state == "S":

                    T[LEFT, this_state_ix, left_state_ix] += 1.0 / 3.0
                    T[LEFT, this_state_ix, up_state_ix] += 1.0 / 3.0
                    T[LEFT, this_state_ix, down_state_ix] += 1.0 / 3.0

                    T[RIGHT, this_state_ix, right_state_ix] += 1.0 / 3.0
                    T[RIGHT, this_state_ix, up_state_ix] += 1.0 / 3.0
                    T[RIGHT, this_state_ix, down_state_ix] += 1.0 / 3.0

                    T[DOWN, this_state_ix, right_state_ix] += 1.0 / 3.0
                    T[DOWN, this_state_ix, left_state_ix] += 1.0 / 3.0
                    T[DOWN, this_state_ix, down_state_ix] += 1.0 / 3.0

                    T[UP, this_state_ix, right_state_ix] += 1.0 / 3.0
                    T[UP, this_state_ix, left_state_ix] += 1.0 / 3.0
                    T[UP, this_state_ix, up_state_ix] += 1.0 / 3.0

                    R[this_state_ix] = 0

                elif current_state == "H":
                    for action in [LEFT, RIGHT, DOWN, UP]:
                        T[action, this_state_ix, this_state_ix] = 1.0
                        R[this_state_ix] = 0
                elif current_state == "G":
                    for action in [LEFT, RIGHT, DOWN, UP]:
                        T[action, this_state_ix, this_state_ix] = 1.0
                        R[this_state_ix] = 1

        return T, R


def render_policy(policy, m):
    s = int(np.sqrt(len(policy)))
    map_str = []
    for i in range(s):
        row_str = ""
        for j in range(s):
            action = policy[i * s + j]
            if m[i][j] == "H":
                row_str += " X "
            elif m[i][j] == "S":
                row_str += " S "
            elif m[i][j] == "G":
                row_str += " G "

            elif action == LEFT:
                row_str += " < "
            elif action == RIGHT:
                row_str += " > "
            elif action == DOWN:
                row_str += " v "
            else:
                row_str += " ^ "
        map_str.append(row_str)
    return map_str
