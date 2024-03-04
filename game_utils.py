from collections import namedtuple
from enum import Enum
import heapq
import torch
import math
import numpy as np


Point = namedtuple('Point', 'x, y')
DEVICE =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

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

def sort_obstacles_with_heapq(snake_head, obstacles):
    # Create a list of (distance, obstacle) tuples
    obstacles_with_distances = [(calculate_distance(snake_head, obstacle), obstacle) for obstacle in obstacles]
    
    # Convert the list into a heap
    heapq.heapify(obstacles_with_distances)
    
    # Initialize an empty list to store sorted obstacles
    sorted_obstacles = []
    
    # Pop elements from the heap until it's empty
    while obstacles_with_distances:
        _, obstacle = heapq.heappop(obstacles_with_distances)
        sorted_obstacles.append(obstacle)
    
    return sorted_obstacles
