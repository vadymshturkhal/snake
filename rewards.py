import numpy as np

from game_settings import REWARD_CRAWLING, REWARD_ROTATION
from game_settings import REWARD_WIN, REWARD_LOOSE

K = 0.001

class Rewards:
    def __init__(self, game):
        self._game = game
        self._transitions = {}
        self._transition_counter = 0

    def get_snake_reward(self, state, action):
        # Goal reached
        if self._game.is_eaten_food:
            return REWARD_WIN

        # Crashed
        if self._game.snake_is_crashed:
            return REWARD_LOOSE

        # Penalty
        if action == (0, 0, 1):
            snake_reward = REWARD_CRAWLING
        else:
            snake_reward = REWARD_ROTATION

        bonus_reward = self._get_bonus(state, action)
        # bonus_reward = 0
        self._store_transition(state, action)
        return snake_reward + bonus_reward

    def _get_bonus(self, state, action):
        tau = self._transition_counter - self._transitions.get((tuple(state), action), 0)
        bonus_reward = K * np.sqrt(tau)
        return bonus_reward

    def _store_transition(self, state, action):
        self._transitions[(tuple(state), action)] = self._transition_counter
        self._transition_counter += 1
