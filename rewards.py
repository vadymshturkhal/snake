import math

from game_utils import calculate_distance, calculate_angle
from game_settings import SCREEN_W, SCREEN_H, SNAKE_ANGLE_PUNISH, SNAKE_ANGLE_REWARD
from game_settings import REWARD_WRONG_DIRECTION, REWARD_CORECT_DIRECTION, REWARD_WIN, REWARD_LOOSE


class Rewards:
    def __init__(self):
        self.max_possible_distance = math.sqrt(SCREEN_W**2 + SCREEN_H**2)
        self.prev_distance = self.max_possible_distance

        self.previous_angle = None

    def get_snake_reward(self, snake, food):
        if self.previous_angle is None:
            self.previous_angle = calculate_angle(snake, food.head)

        # Assuming snake_head and food_position are Point objects with x and y attributes
        distance = calculate_distance(snake.head, food.head)
        current_angle = calculate_angle(snake, food.head)

        snake_reward = 0
        if current_angle <= self.previous_angle:
            # Snake is turning towards the food
            snake_reward += SNAKE_ANGLE_REWARD
        else:
            # Snake is turning away from the food
            snake_reward += SNAKE_ANGLE_PUNISH

        self.previous_angle = current_angle

        if self.prev_distance > distance:
            snake_reward += REWARD_CORECT_DIRECTION
        else:
            snake_reward += REWARD_WRONG_DIRECTION

        self.prev_distance = distance

        return snake_reward
