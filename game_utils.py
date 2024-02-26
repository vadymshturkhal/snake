from collections import namedtuple
from enum import Enum
import torch
import math


Point = namedtuple('Point', 'x, y')
DEVICE =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

def calculate_distance_and_angle(snake_head, food):
    dx = food.x - snake_head.x
    dy = food.y - snake_head.y
    distance = math.sqrt(dx**2 + dy**2)
    angle = math.atan2(dy, dx)
    return distance, angle

def normalize_distance(distance, max_distance):
    return distance / max_distance