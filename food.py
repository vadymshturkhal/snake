import random
from game_utils import Direction, Point
from game_settings import BLOCK_SIZE


class Food:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.coordinates = Point(x, y)
