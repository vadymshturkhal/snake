import math

from game_utils import calculate_distance
from game_settings import SCREEN_W, SCREEN_H
from game_settings import REWARD_WRONG_DIRECTION, REWARD_CORECT_DIRECTION, REWARD_WIN, REWARD_LOOSE
PENALTY_FOR_TIME_WASTING = -0.01


class Rewards:
    def __init__(self, game):
        self.snake = game.snake
        self.food = game.food

        self.max_possible_distance = math.sqrt(SCREEN_W**2 + SCREEN_H**2)
        self.prev_distance = self.max_possible_distance

    def get_snake_reward(self):
        # Assuming snake_head and food_position are Point objects with x and y attributes
        distance = calculate_distance(self.snake.head, self.food.head)

        if self.prev_distance > distance:
            snake_reward = REWARD_CORECT_DIRECTION
        else:
            snake_reward = REWARD_WRONG_DIRECTION
        
        snake_reward += PENALTY_FOR_TIME_WASTING

        self.prev_distance = distance

        return snake_reward