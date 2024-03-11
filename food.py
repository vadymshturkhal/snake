import random
import numpy as np
from game_utils import Point, Direction
from game_settings import DIRECTIONS_QUANTITY, BLOCK_SIZE, SCREEN_W, SCREEN_H


class Food:
    def __init__(self, position):
        self.position = position
        self.direction = random.choice([Direction.RIGHT, Direction.LEFT, Direction.UP, Direction.DOWN])

    def move(self, action=None):
        # [straight, right, left, no action]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]

        # If action is None, choose a random action
        if action is None:
            action = [0, 0, 0, 0]
            action[random.randint(0, 3)] = 1  # Set one of the actions to 1 at random

        # Current direction index
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0, 0]):
            new_dir = clock_wise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0, 0]):
            next_idx = (idx + 1) % DIRECTIONS_QUANTITY
            new_dir = clock_wise[next_idx]  # right turn r -> d -> l -> u
        elif np.array_equal(action, [0, 0, 1, 0]):
            next_idx = (idx - 1) % DIRECTIONS_QUANTITY
            new_dir = clock_wise[next_idx]  # left turn r -> u -> l -> d
        elif np.array_equal(action, [0, 0, 0, 1]):
            new_dir = self.direction  # No action
        else:
            raise Exception('Unknown direction')

        self.direction = new_dir

        x = self.position.x
        y = self.position.y
        if self.direction == Direction.RIGHT:
            x = min(x + BLOCK_SIZE, SCREEN_W - BLOCK_SIZE)
        elif self.direction == Direction.LEFT:
            x = max(x - BLOCK_SIZE, 0)
        elif self.direction == Direction.DOWN:
            y = min(y + BLOCK_SIZE, SCREEN_H - BLOCK_SIZE)
        elif self.direction == Direction.UP:
            y = max(y - BLOCK_SIZE, 0)

        self.position = Point(x, y)
