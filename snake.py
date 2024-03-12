import random
import numpy as np
from game_utils import Point, Direction
from game_settings import DIRECTIONS_QUANTITY, BLOCK_SIZE, SCREEN_W, SCREEN_H


class Snake:
    def __init__(self, head, init_direction=None):
        self.head = head
        self.direction = init_direction

        if self.direction is None:
            self.direction = random.choice([Direction.RIGHT, Direction.LEFT, Direction.UP, Direction.DOWN])

    def move(self, action, is_human=False) -> bool:
        if is_human:
            self.direction = action
        else:
            # [straight, right, left]
            clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]

            # Current direction index
            idx = clock_wise.index(self.direction)

            if np.array_equal(action, [1, 0, 0]):
                new_dir = clock_wise[idx] # no change
            elif np.array_equal(action, [0, 1, 0]):
                next_idx = (idx + 1) % DIRECTIONS_QUANTITY
                new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
            elif np.array_equal(action, [0, 0, 1]): # [0, 0, 1]
                next_idx = (idx - 1) % DIRECTIONS_QUANTITY
                new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d
            else:
                raise Exception('Unknown direction', )

            self.direction = new_dir


        x = self.head.x
        y = self.head.y

        is_crashed = False

        if self.direction == Direction.RIGHT:
            x_new = x + BLOCK_SIZE
            if x_new < SCREEN_W:
                x = x_new
            else:
                is_crashed = True

        elif self.direction == Direction.LEFT:
            x_new = x - BLOCK_SIZE
            if x_new >= 0:
                x = x_new
            else:
                is_crashed = True

        elif self.direction == Direction.DOWN:
            y_new = y + BLOCK_SIZE
            if y_new < SCREEN_H:
                y = y_new
            else:
                is_crashed = True

        elif self.direction == Direction.UP:
            y_new = y - BLOCK_SIZE
            if y_new >= 0:
                y = y_new
            else:
                is_crashed = True

        self.head = Point(x, y)

        return is_crashed
