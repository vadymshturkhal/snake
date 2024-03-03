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

def calculate_distance(snake_head, food_head):
    dx = food_head.x - snake_head.x
    dy = food_head.y - snake_head.y
    return math.sqrt(dx**2 + dy**2)

def calculate_angle(snake, food_position):
    snake_head = snake.head
    vector_to_food = (food_position.x - snake_head.x, food_position.y - snake_head.y)
    direction_vector = {Direction.UP: (0, -1), Direction.DOWN: (0, 1),
                        Direction.LEFT: (-1, 0), Direction.RIGHT: (1, 0)}[snake.direction]

    dot_product = vector_to_food[0] * direction_vector[0] + vector_to_food[1] * direction_vector[1]
    magnitude_vector_to_food = math.sqrt(vector_to_food[0]**2 + vector_to_food[1]**2)
    magnitude_direction_vector = 1  # Direction vectors are unit vectors

    # Avoid division by zero by ensuring neither magnitude is zero
    if magnitude_vector_to_food == 0 or magnitude_direction_vector == 0:
        return 0  # Can decide on a default value or handling mechanism

    # Proceed with angle calculation
    cos_angle = dot_product / (magnitude_vector_to_food * magnitude_direction_vector)
    
    # Clamp the cos_angle to the range [-1, 1] to avoid math domain errors
    cos_angle = max(min(cos_angle, 1), -1)
    
    angle = math.acos(cos_angle)
    
    # Convert angle to degrees for easier interpretation
    angle_degrees = math.degrees(angle)
    
    return angle_degrees

def normalize_distance(distance, max_distance):
    return distance / max_distance