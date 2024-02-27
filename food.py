import random
from game_utils import Direction, Point, Direction
from game_settings import BLOCK_SIZE, SCREEN_W, SCREEN_H


class Food:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.coordinates = Point(x, y)
        self.direction = None

    def move(self):
        directions = [Direction.RIGHT, Direction.LEFT, Direction.UP, Direction.DOWN]

        # Randomly change direction
        self.direction = random.choice(directions)
        # self.direction = Direction.RIGHT
        
        x, y = self.x, self.y
        if self.direction == Direction.RIGHT:
            x = min(x + BLOCK_SIZE, SCREEN_W - BLOCK_SIZE)
        elif self.direction == Direction.LEFT:
            x = max(x - BLOCK_SIZE, 0)
        elif self.direction == Direction.UP:
            y = max(y - BLOCK_SIZE, 0)
        elif self.direction == Direction.DOWN:
            y = min(y + BLOCK_SIZE, SCREEN_H - BLOCK_SIZE)
        
        self._update_coordinates(x, y)

    def _update_coordinates(self, x, y):
        self.x = x
        self.y = y
        self.coordinates = Point(x, y)