import gym
from gym import utils
import sys
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
    "SFHFFFFFFFFFFFHFHFFFFFFFFFHFFFFF",
    "FFFFFFHFFFFFFFFFHFFHFFFFFFFFFFHF",
    "HFFFFFFFHFFFFFHHFHFFFFFFFFFFFFHF",
    "FFFFFFFFFFFFFFHFFFFFFFFHFFFHHFFF",
    "HFFFFHFFFFFFHFHFFFHFHFFFFFFFHFFF",
    "HFFFFFFHFFHFFFFFFFFFFFFFFFFFFFFF",
    "FFFFFFFFFFFHHHFFHFFFFFFFFFFFFFHF",
    "FFFFFFFHHFFFFFHFFFFFFFFFFHFFFHFF",
    "FFFFFFFFFFFFHFFFFHFFFHFFFFFFFFHH",
    "HFFFFFFFFFFFFFFFHFHFFFFFFFFFFFHF",
    "HFFFFFFFFFFFFHFFHFFHFFFFFFFFHHHF",
    "FFFHFFFHFFHFFFFFFFFFHFFFHFFHFFFF",
    "FFFFFFFFFFFFFFFFFFHFHFFFFFFFFFFF",
    "FFFFFFFFFFFHFFFFFHFFFFFFFHHFFHHF",
    "FFHFFHFFFFFFFFFFFFFFFHFFFHFHFHHF",
    "FFFFFFFFFFFFFFFFFFFFFFFHHFFHFFFF",
    "HFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
    "FFFFFFFFFFFFFFFFFFFFFFHFFFFHFFFF",
    "FFFFHFFFFFFFFFFFFFFFFFHHFHFFFHFF",
    "FFFFFFFFFFFFHHHFFHFFFHFFFFFFFFFF",
    "FFFHFFHFHFFFFFFHFFFFFFFFHFFFFFHF",
    "FFHFFFHFHFFFFFFFFFFFFFFFFFFFHFFH",
    "HHFFFFFFFFFFHFHFFHFHHFFFFFFFFFFF",
    "FHFFFHFFFFFFFFFFFFFFFFFFFFFFFFFF",
    "HFFFHFFFFFHFFFFFFFFHFFHFHFFFFFFF",
    "FHFHFFHFFHFFFFHFFFFHFFHFFFHFHFFF",
    "FFFFFHFFFFFFFFFFHFFFFFFFFFFFFHFF",
    "FHFFFFFFFFFFFFFFFFFFHFFHHFHHHFFF",
    "FFFFFHFFFFFHFFFFHFFFFFFFFFFFFFFF",
    "FFFFFFFFFFFFFFFFFFHFFHFHFFFFFFFH",
    "HHFFFFHFFFFFFFFFFFFHFHFFFHFFFHFH",
    "FFFFFFFFFFFFFHFFFFFFFFHFFFFFFFFG",
]

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

# utility functions
def get_small_lake():
    return FrozenLake(m_small)


def get_large_lake():
    return FrozenLake(m_large)


class FrozenLake:
    def __init__(self, m):
        self.m = m
        self.rows = len(self.m)
        self.cols = self.rows  # only support square grids
        self.pos = 0

    def move_left(self, pos):
        row = pos // self.rows
        col = pos % self.cols

        if col - 1 < 0:
            return pos

        else:
            return row * self.rows + (col - 1)

    def move_right(self, pos):
        row = pos // self.rows
        col = pos % self.cols

        if col + 1 >= self.cols:
            return pos  # bounce off wall
        else:
            return row * self.rows + col + 1

    def move_up(self, pos):
        row = pos // self.rows
        col = pos % self.cols

        if row - 1 < 0:
            return pos  # bounce
        else:
            return (row - 1) * self.rows + col

    def move_down(self, pos):
        row = pos // self.rows
        col = pos % self.cols

        if row + 1 >= self.rows:
            return pos  # bounce
        else:
            return (row + 1) * self.rows + col

    def reset(self):
        self.pos = 0
        return self.pos

    def step(self, action):
        assert action in [0, 1, 2, 3], f"Invalid action: {action}"

        assert not self.get_pos_type(self.pos) in [
            "G",
            "H",
        ], "Cannot step in terminal state"
        row = self.pos // self.rows

        if action == LEFT:
            left_state_pos = self.move_left(self.pos)
            up_state_pos = self.move_up(self.pos)
            down_state_pos = self.move_down(self.pos)
            new_pos = np.random.choice(
                [left_state_pos, up_state_pos, down_state_pos], p=[1.0 / 3.0] * 3
            )

        elif action == RIGHT:
            right_state_pos = self.move_right(self.pos)
            up_state_pos = self.move_up(self.pos)
            down_state_pos = self.move_down(self.pos)
            new_pos = np.random.choice(
                [right_state_pos, up_state_pos, down_state_pos], p=[1.0 / 3.0] * 3
            )

        elif action == DOWN:
            right_state_pos = self.move_right(self.pos)
            left_state_pos = self.move_left(self.pos)
            down_state_pos = self.move_down(self.pos)

            new_pos = np.random.choice(
                [right_state_pos, left_state_pos, down_state_pos], p=[1.0 / 3.0] * 3
            )

        elif action == UP:
            right_state_pos = self.move_right(self.pos)
            up_state_pos = self.move_up(self.pos)
            left_state_pos = self.move_left(self.pos)
            new_pos = np.random.choice(
                [right_state_pos, up_state_pos, left_state_pos], p=[1.0 / 3.0] * 3
            )

        next_pos_type = self.get_pos_type(new_pos)
        if next_pos_type == "G":
            reward = 1
            terminal = True
        elif next_pos_type == "H":
            reward = 0
            terminal = True
        else:
            reward = 0
            terminal = False

        self.pos = new_pos
        return new_pos, reward, terminal

    def get_pos_type(self, pos):
        row = pos // self.rows
        col = pos % self.cols
        return self.m[row][col]

    @property
    def action_space(self):
        return [0, 1, 2, 3]

    @property
    def state_space(self):
        return len(self.m) ** 2

    def build_TR_matrices(self):
        # Should generate the appropriate transition and reward matrices for any given map
        m = self.m  # alias
        size = len(m)
        T = np.zeros((4, size * size, size * size))
        R = np.zeros((4, size * size, size * size))

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

                    # Ending up in this state provides a reward of zero
                    R[:, :, this_state_ix] = 0

                elif current_state == "H":
                    for action in [LEFT, RIGHT, DOWN, UP]:
                        T[action, this_state_ix, this_state_ix] = 1.0
                        R[:, :, this_state_ix] = 0
                elif current_state == "G":
                    for action in [LEFT, RIGHT, DOWN, UP]:
                        T[action, this_state_ix, this_state_ix] = 1.0
                        R[:, :, this_state_ix] = 1.0

        return T, R

    def render(self):
        """Mostly taken from https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py

        I mostly built the frozen lake myself, but used their renderer and map generator"""
        outfile = sys.stdout

        row, col = self.pos // self.rows, self.pos % self.rows
        desc = np.asarray(self.m, dtype="c")
        desc = [[c.decode("utf-8") for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        outfile.write("\n")
        outfile.write("\n".join("".join(line) for line in desc) + "\n")


def create_frozenlake_policy(raw_policy):
    """Accepts a policy as given by pymdptoolbox and turns it into a function readable by an Agent"""

    def policy(state):
        return raw_policy[state]

    return policy
