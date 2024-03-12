import pygame
import random
import numpy as np
from game_utils import Point, Direction
from game_settings import DIRECTIONS_QUANTITY, BLOCK_SIZE, SCREEN_W, SCREEN_H, SNAKE_SPRITE_PATH


class Snake:
    def __init__(self, head, init_direction=None):
        self.head = head
        self.direction = init_direction

        # Load the snake head sprite
        self.sprite = pygame.image.load(SNAKE_SPRITE_PATH)
        self.sprite = pygame.transform.scale(self.sprite, (BLOCK_SIZE, BLOCK_SIZE))
        self.sprite_rotated = self.sprite

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

        self._rotate_sprite()
        return is_crashed

    def _rotate_sprite(self):
        print(self.direction)
        if self.direction == Direction.UP:
            rotated_sprite = pygame.transform.rotate(self.sprite, 0)  # Assumes the original sprite is facing up
        elif self.direction == Direction.RIGHT:
            rotated_sprite = pygame.transform.rotate(self.sprite, 270)  # Rotate clockwise
        elif self.direction == Direction.DOWN:
            rotated_sprite = pygame.transform.rotate(self.sprite, 180)
        else:  # self.direction == Direction.LEFT
            rotated_sprite = pygame.transform.rotate(self.sprite, 90)

        self.sprite_rotated = rotated_sprite
