from collections import namedtuple
from enum import Enum
import torch
import numpy as np
from game_settings import BLOCK_SIZE, SCREEN_W, SCREEN_H


Point = namedtuple('Point', 'x, y')
DEVICE =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)
    UP_LEFT = (-1, -1)
    UP_RIGHT = (1, -1)
    DOWN_LEFT = (-1, 1)
    DOWN_RIGHT = (1, 1)

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

def calculate_distance(snake_head, food_head):
    # dx = food_head.x - snake_head.x
    # dy = food_head.y - snake_head.y
    # return math.sqrt(dx**2 + dy**2)

    # Manhattan
    dx = abs(food_head.x - snake_head.x)
    dy = abs(food_head.y - snake_head.y)
    return dx + dy

def calculate_angle(snake, food_position):
    snake_head = snake.head
    # Convert enum to direction vector
    snake_direction_vector = np.array(snake.direction.value)
    
    # Vector from snake to food
    vector_to_food = np.array([food_position[0] - snake_head[0], food_position[1] - snake_head[1]])
    
    # Normalize the vector to food to have a magnitude of 1 for accurate angle calculation
    norm = np.linalg.norm(vector_to_food)
    if norm == 0:
        #  Depending on your game logic, you might want to set vector_to_food_normalized to a specific value when the snake is at the food position
        vector_to_food_normalized = vector_to_food  # Or handle as appropriate for your application
    else:
        vector_to_food_normalized = vector_to_food / norm
    
    # Calculate the dot product and angle
    dot_product = np.dot(vector_to_food_normalized, snake_direction_vector)
    dot_product = np.clip(dot_product, -1.0, 1.0)  # Ensure it's within the valid range for arccos
    angle = np.arccos(dot_product)
    
    # Convert angle from radians to degrees
    angle_degrees = np.degrees(angle)
    
    return angle_degrees

def normalize_distance(distance, max_distance):
    return distance / max_distance

def sort_obstacles(snake_head, obstacles):
    # Calculate distance from snake head to each obstacle and store in a list of tuples
    distances = [(obstacle, calculate_distance(snake_head, obstacle)) for obstacle in obstacles]
    
    # Sort the list of tuples based on distance
    sorted_obstacles = sorted(distances, key=lambda x: x[1])
    
    # Extract sorted obstacles without distances
    sorted_obstacles_without_distances = [obstacle[0] for obstacle in sorted_obstacles]
    
    return sorted_obstacles

def ray_trace_to_obstacle(head, direction, obstacles):
    """
    Trace a ray from the snake's head in the current direction until it hits an obstacle or boundary.
    
    :param head: The starting point of the ray (snake's head position).
    :param direction: The current direction of the snake.
    :param obstacles: A list of obstacle positions.
    :return: The distance to the closest obstacle in the current direction.
    """
    BLOCK_SIZE = 20  # Assuming a defined block size for movements.
    dx, dy = 0, 0

    # Determine the direction vector
    if direction == Direction.RIGHT:
        dx = BLOCK_SIZE
    elif direction == Direction.LEFT:
        dx = -BLOCK_SIZE
    elif direction == Direction.UP:
        dy = -BLOCK_SIZE
    elif direction == Direction.DOWN:
        dy = BLOCK_SIZE

    distance = 0
    current_position = Point(head.x, head.y)

    # Move the ray step by step in the specified direction
    while True:
        current_position = Point(current_position.x + dx, current_position.y + dy)
        distance += BLOCK_SIZE

        # Check for boundary collision
        if current_position.x < 0 or current_position.x >= SCREEN_W or current_position.y < 0 or current_position.y >= SCREEN_H:
            break

        # Check for obstacle collision
        if current_position in obstacles:
            break

    return distance

def check_dangers(game):
    dangers = []
    for direction in Direction:
        dx, dy = direction.value
        new_x = game.snake.head.x + (dx * BLOCK_SIZE)
        new_y = game.snake.head.y + (dy * BLOCK_SIZE)
        nearest_point = Point(new_x, new_y)

        # Check if the new position is within the grid boundaries
        if game.is_collision(nearest_point):
            dangers.append(True)
        else:
            dangers.append(False)
    
    return dangers
