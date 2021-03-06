import numpy as np

HUNT = 0
WAIT = 1

ANIMAL_IX = 0
ENERGY_IX = 1
INJURED_IX = 2

NO_ANIMAL = -1

STARVED = -1


class TerminalStateError(Exception):
    pass


class Animal:
    def __init__(self, name, energy, escape_chance, injure_chance):
        assert (
            escape_chance + injure_chance <= 1
        ), "Escape and injury chance can't cumulatively exceed 100% likelihood"
        self.name = name
        self.energy = energy
        self.escape_chance = escape_chance
        self.injury_chance = injure_chance


class HuntingMDP:
    def __init__(
        self,
        animals,
        appearance_weights,
        hunting_cost=4,
        living_cost=2,
        recovery_time=5,
        injured_penalty=0.1,
        max_energy=50
    ):
        assert sum(appearance_weights) <= 1, "Appearance weights cannot exceed 1"
        self.appearance_weights = {}

        self.N_animals = len(animals)
        for i in range(len(animals)):
            self.appearance_weights[i] = appearance_weights[i]

        self.appearance_weights[NO_ANIMAL] = 1 - sum(appearance_weights)

        self.max_energy = max_energy

        # normalize appearance weights

        self.hunting_cost = hunting_cost
        self.living_cost = living_cost
        self.recovery_time = recovery_time
        self.injured_penalty = injured_penalty

        self.animals = animals

        self.build_state_conversion_dicts()

        state = []

    def get_action_space(self, state):
        animal_present = state[ANIMAL_IX] != NO_ANIMAL
        if state[INJURED_IX] > 0 or not animal_present:
            return [WAIT]
        else:
            return [HUNT, WAIT]

    def reset(self):
        # Start state will always be max energy, random animal
        state = (np.random.randint(-1, self.N_animals), self.max_energy, 0)
        self.state = state
        return state

    def is_terminal(self, state):
        if state[ENERGY_IX] <= 0:
            return True
        else:
            return False

    def get_reward(self, state):
        if not self.is_terminal(state):
            reward = 1
        else:
            reward = 0

        return reward

    @property
    def state_space(self):
        return len(self.state_to_idx)

    @property
    def action_space(self):
        return [0, 1]

    def step(self, action):
        state = self.state  # alias
        if self.is_terminal(state):
            raise TerminalStateError("Cannot perform step on terminal state")

        animal = state[ANIMAL_IX]
        injury = state[INJURED_IX]
        next_animal = np.random.choice(
            list(self.appearance_weights.keys()),
            p=list(self.appearance_weights.values()),
        )
        if action == HUNT:
            escape_chance = self.animals[animal].escape_chance
            injury_chance = self.animals[animal].injury_chance

            success_chance = 1 - escape_chance - injury_chance
            # Normalize by injury penalty
            escape_chance = escape_chance * (self.injured_penalty * injury + 1)
            injury_chance = injury_chance * (self.injured_penalty * injury + 1)

            normalization_constant = success_chance + escape_chance + injury_chance
            success_chance /= normalization_constant
            escape_chance /= normalization_constant
            injury_chance /= normalization_constant

            hunt_result = np.random.choice(
                ["success", "escape", "injury"],
                p=[success_chance, escape_chance, injury_chance],
            )
            result = self.get_result_state(
                state, action, next_animal, result_type=hunt_result
            )
        else:
            result = self.get_result_state(state, action, next_animal)

        reward = self.get_reward(result)
        terminal = self.is_terminal(result)

        self.state = result
        return result, reward, terminal

    def get_result_state(self, state, action, next_animal, result_type="success"):
        assert result_type in ["success", "escape", "injury"], "Invalid result type"
        if isinstance(state, list):
            result_state = state.copy()
        else:
            result_state = list(state)

        injured = state[INJURED_IX]
        animal = state[ANIMAL_IX]

        # Handle the no-animal case
        if animal == NO_ANIMAL and action == HUNT:
            # Always fail, no matter what
            result_state[ENERGY_IX] -= self.hunting_cost + self.living_cost
        else:
            if action == HUNT and result_type == "success":
                result_state[ENERGY_IX] -= self.hunting_cost + self.living_cost
                result_state[ENERGY_IX] += self.animals[animal].energy
            elif action == HUNT and result_type == "escape":
                result_state[ENERGY_IX] -= self.hunting_cost + self.living_cost
            elif action == HUNT and result_type == "injury":
                result_state[ENERGY_IX] -= self.hunting_cost + self.living_cost
                result_state[INJURED_IX] = self.recovery_time

            elif action == WAIT:
                result_state[ENERGY_IX] -= self.living_cost
                if injured > 0:
                    result_state[INJURED_IX] -= 1
            else:
                raise NotImplementedError("I don't htink this should be possible")

        # truncate energy
        if result_state[ENERGY_IX] < 0:
            result_state[ENERGY_IX] = STARVED
        elif result_state[ENERGY_IX] > self.max_energy:
            result_state[ENERGY_IX] = self.max_energy

        result_state[ANIMAL_IX] = next_animal
        return tuple(result_state)

    def build_state_conversion_dicts(self):
        self.state_to_idx = {}
        self.idx_to_state = {}
        ix = 0
        for animal_ix in range(-1, self.N_animals):
            for energy in range(-1, self.max_energy + 1):
                for injury in range(0, self.recovery_time + 1):
                    state = (animal_ix, energy, injury)
                    self.state_to_idx[state] = ix
                    self.idx_to_state[ix] = state
                    ix += 1

    def build_TR_matrices(self):
        i = 0

        N_states = len(self.state_to_idx)
        T = np.zeros((2, N_states, N_states))
        R = np.zeros((2, N_states, N_states))
        for action in range(2):
            for animal_ix in range(-1, self.N_animals):
                for energy in range(-1, self.max_energy + 1):
                    for injury in range(0, self.recovery_time + 1):
                        state = (animal_ix, energy, injury)
                        state_ix = self.state_to_idx[state]
                        if self.is_terminal(state):
                            T[action, state_ix, state_ix] = 1
                            R[
                                action, state_ix, state_ix
                            ] = 0  

                        else:
                            for next_animal in range(-1, self.N_animals):
                                if action == HUNT:
                                    result_state_success = self.get_result_state(
                                        state,
                                        action,
                                        next_animal,
                                        result_type="success",
                                    )
                                    result_state_escape = self.get_result_state(
                                        state, action, next_animal, result_type="escape"
                                    )
                                    result_state_injury = self.get_result_state(
                                        state, action, next_animal, result_type="injury"
                                    )

                                    success_ix = self.state_to_idx[result_state_success]
                                    escape_ix = self.state_to_idx[result_state_escape]
                                    injury_ix = self.state_to_idx[result_state_injury]

                                    escape_chance = self.animals[
                                        animal_ix
                                    ].escape_chance
                                    injury_chance = self.animals[
                                        animal_ix
                                    ].injury_chance

                                    success_chance = 1 - escape_chance - injury_chance
                                    # Normalize by injury penalty
                                    escape_chance = escape_chance * (
                                        self.injured_penalty * injury + 1
                                    )
                                    injury_chance = injury_chance * (
                                        self.injured_penalty * injury + 1
                                    )

                                    normalization_constant = (
                                        success_chance + escape_chance + injury_chance
                                    )
                                    success_chance /= normalization_constant
                                    escape_chance /= normalization_constant
                                    injury_chance /= normalization_constant

                                    T[action, state_ix, escape_ix] += (
                                        escape_chance
                                        * self.appearance_weights[next_animal]
                                    )
                                    T[action, state_ix, injury_ix] += (
                                        injury_chance
                                        * self.appearance_weights[next_animal]
                                    )
                                    T[action, state_ix, success_ix] += (
                                        success_chance
                                        * self.appearance_weights[next_animal]
                                    )

                                    R[action, state_ix, success_ix] = self.get_reward(
                                        result_state_success
                                    )
                                    R[action, state_ix, injury_ix] = self.get_reward(
                                        result_state_injury
                                    )
                                    R[action, state_ix, escape_ix] = self.get_reward(
                                        result_state_escape
                                    )

                                else:
                                    result_state = self.get_result_state(
                                        state, action, next_animal
                                    )
                                    result_ix = self.state_to_idx[result_state]
                                    T[
                                        action, state_ix, result_ix
                                    ] += self.appearance_weights[next_animal]
                                    R[action, state_ix, result_ix] = self.get_reward(
                                        result_state
                                    )
        return T, R
