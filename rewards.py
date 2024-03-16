import math

from game_utils import calculate_distance, calculate_angle, check_dangers
from game_settings import EPSILON_SHIFT, REWARD_CRAWLING, REWARD_ROTATION, SNAKE_ANGLE_PUNISH, SNAKE_ANGLE_REWARD
from game_settings import REWARD_WRONG_DIRECTION, REWARD_CORECT_DIRECTION, REWARD_WIN, REWARD_LOOSE


class Rewards:
    def __init__(self, game):
        self.game = game
        self.prev_distance = game.max_possible_distance
        self.previous_angle = None

    def get_snake_reward(self, action):
        # Goal reached
        if self.game.is_eaten():
            return REWARD_WIN

        # Crashed
        if self.game.snake_is_crashed:
            return REWARD_LOOSE

        if action == [0, 0, 1]:
            snake_reward = REWARD_CRAWLING
        else:
            snake_reward = REWARD_ROTATION


        return snake_reward
