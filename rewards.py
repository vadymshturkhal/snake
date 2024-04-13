from game_settings import REWARD_CRAWLING, REWARD_ROTATION
from game_settings import REWARD_WIN, REWARD_LOOSE


class Rewards:
    def __init__(self, game):
        self._game = game

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

        return snake_reward
