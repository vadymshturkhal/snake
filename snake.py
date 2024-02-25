import random
import numpy as np
from game_utils import Point, Direction


DIRECTIONS_QUANTITY = 4
BLOCK_SIZE = 20


class Snake:
        def __init__(self, head, init_direction=None):
            self.head = head
            self.direction = init_direction

            if self.direction is None:
                self.direction = random.choice([Direction.RIGHT, Direction.LEFT, Direction.UP, Direction.DOWN])

        def _move(self, action):
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
            if self.direction == Direction.RIGHT:
                x += BLOCK_SIZE
            elif self.direction == Direction.LEFT:
                x -= BLOCK_SIZE
            elif self.direction == Direction.DOWN:
                y += BLOCK_SIZE
            elif self.direction == Direction.UP:
                y -= BLOCK_SIZE

            self.head = Point(x, y)

        def reset(self):
            # init game state
            self.direction = Direction.RIGHT

            self.head = Point(self.w/2, self.h/2)
            self.snake = [self.head]

            self.score = 0
            self.food = None
            self._place_food()
            self.frame_iteration = 0

            self.food_direction = random.choice([Direction.RIGHT, Direction.LEFT, Direction.UP, Direction.DOWN])
            # self._move_food()