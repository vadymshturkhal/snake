import pygame
import random
import numpy as np
from game_utils import CLOCK_WISE, Point, Direction
from game_settings import DIRECTIONS_QUANTITY, BLOCK_SIZE, SCREEN_W, SCREEN_H, SNAKE_SPRITE_PATH


class Snake:
    def __init__(self, head, game, init_direction=None):
        self.head = head
        self.game = game
        self.direction = init_direction
        self.prev_direction = self.direction

        # Load the snake head sprite
        self.sprite = pygame.image.load(SNAKE_SPRITE_PATH)
        self.sprite = pygame.transform.scale(self.sprite, (BLOCK_SIZE, BLOCK_SIZE))
        self.sprite_rotated = self.sprite
        self._rotate_sprite()

        if self.direction is None:
            self.direction = random.choice([Direction.RIGHT, Direction.LEFT, Direction.UP, Direction.DOWN])

    def rotate_snake(self, desired_direction: str):
        if desired_direction == 'left':
            # Rotate left
            idx = CLOCK_WISE.index(self.direction)
            next_idx = (idx - 1) % DIRECTIONS_QUANTITY
            next_direction = CLOCK_WISE[next_idx]
        elif desired_direction == 'right':
            # Rotate right
            idx = CLOCK_WISE.index(self.direction)
            next_idx = (idx + 1) % DIRECTIONS_QUANTITY
            next_direction = CLOCK_WISE[next_idx]
        else:
            print('Unknown desired direction')
            raise Exception('rotate_snake unknown desired_direction')

        self.direction = next_direction
        self._rotate_sprite()

    def move_after_rotation(self):
        x = self.head.x
        y = self.head.y

        is_crashed = False

        # Temporary variables to hold the new potential position
        x_new = x
        y_new = y

        if self.direction == Direction.RIGHT:
            x_new = x + BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x_new = x - BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y_new = y + BLOCK_SIZE
        elif self.direction == Direction.UP:
            y_new = y - BLOCK_SIZE

        # Check boundary collision
        if x_new >= SCREEN_W or x_new < 0 or y_new >= SCREEN_H or y_new < 0:
            is_crashed = True
        # Check obstacle collision
        elif any(obstacle.x == x_new and obstacle.y == y_new for obstacle in self.game.obstacles):
            is_crashed = True
        else:
            # Update the head position only if there's no crash
            self.head = Point(x_new, y_new)

        return is_crashed

    def _rotate_sprite(self):
        if self.direction == Direction.UP:
            rotated_sprite = pygame.transform.rotate(self.sprite, 0)  # Assumes the original sprite is facing up
        elif self.direction == Direction.RIGHT:
            rotated_sprite = pygame.transform.rotate(self.sprite, 270)  # Rotate clockwise
        elif self.direction == Direction.DOWN:
            rotated_sprite = pygame.transform.rotate(self.sprite, 180)
        else:  # self.direction == Direction.LEFT
            rotated_sprite = pygame.transform.rotate(self.sprite, 90)

        self.sprite_rotated = rotated_sprite
