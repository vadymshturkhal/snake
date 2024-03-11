import torch
import random
import numpy as np
from collections import deque
from model import Linear_QNet, QTrainer

from game_utils import Direction, Point, calculate_distance, check_dangers, normalize_distance, calculate_angle, ray_trace_to_obstacle
from game_settings import EPSILON_SHIFT, MAX_MEMORY, BATCH_SIZE, LR, AVAILABLE_SNAKE_DIRECTIONS_QUANTITY, BLOCK_SIZE, SNAKE_GAMMA, SNAKE_MIN_EPSILON
from game_settings import SNAKE_INPUT_LAYER_SIZE, SNAKE_HIDDEN_LAYER_SIZE1, SNAKE_HIDDEN_LAYER_SIZE2, SNAKE_OUTPUT_LAYER_SIZE


class SnakeAgent:
    def __init__(self, is_load_weights=False, weights_filename=None, epochs=100):
        self.epsilon = 1
        self.epochs = epochs

        self.gamma = SNAKE_GAMMA
        self.memory = deque(maxlen=MAX_MEMORY)

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = Linear_QNet(SNAKE_INPUT_LAYER_SIZE, SNAKE_HIDDEN_LAYER_SIZE1, SNAKE_HIDDEN_LAYER_SIZE2, SNAKE_OUTPUT_LAYER_SIZE)
        self.model.to(self.device)

        # Load the weights onto the CPU or GPU
        if is_load_weights:
            checkpoint = torch.load(weights_filename, map_location=self.device)
            checkpoint = torch.load(weights_filename, self.device)

            self.n_games = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
        else:
            self.model.to(self.device)
            self.n_games = 0

        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_vision_based_state(self, game, vision_range=1):
        """
        Generate a vision-based state representation around the snake's head.
    
        The function creates a grid centered around the snake's head, with each cell
        representing different entities in the game (empty space, food, snake body, or wall).
        This grid is then flattened into a vector to serve as the state representation
        for input into a machine learning model.

        Parameters:
            - game: A game instance containing the current state of the game,
                    including the snake's position, the food's position, and game dimensions.

            - vision_range: How far the snake can see in each direction.

        Returns:
        - state_vector: A flattened numpy array representing the contents of the grid
                        around the snake's head.
        """
        grid_size = 2 * vision_range + 1  # Calculate the size of the vision grid
        state_grid = np.zeros((grid_size, grid_size))  # Initialize the grid to zeros

        # Define the grid's center point (the snake's head position)
        center_x, center_y = game.snake.head.x, game.snake.head.y

        # Populate the grid
        for i in range(-vision_range, vision_range + 1):
            for j in range(-vision_range, vision_range + 1):
                point = Point(center_x + i * BLOCK_SIZE, center_y + j * BLOCK_SIZE)
                
                # Check if the point is outside the game boundaries (a wall)
                if point.x < 0 or point.y < 0 or point.x >= game.width or point.y >= game.height:
                    state_grid[i + vision_range, j + vision_range] = 2
                # Check if the point is the location of the food
                elif point == game.food.position:
                    state_grid[i + vision_range, j + vision_range] = 1

        # Flatten the grid to create a state vector or use as is for CNN input
        state_vector = state_grid.flatten()
        
        return state_vector

    def get_state(self, game):
        head = game.snake.head

        snake_vision = self.get_vision_based_state(game)

        food_left = game.food.position.x < head.x
        food_right = game.food.position.x > head.x
        food_above = game.food.position.y < head.y
        food_below = game.food.position.y > head.y

        moving_left = game.snake.direction == Direction.LEFT
        moving_right = game.snake.direction == Direction.RIGHT
        moving_up = game.snake.direction == Direction.UP
        moving_down = game.snake.direction == Direction.DOWN

        # Assuming snake_head and food_position are Point objects with x and y attributes
        # distance = calculate_distance(head, game.food.position)
        # angle = calculate_angle(game.snake, game.food.position)
        # normalized_angle = angle / 360  # Example normalization if angle is in degrees
        # normalized_distance = normalize_distance(distance, game.max_possible_distance)

        # Add ray tracing distance to the state
        # distance_to_obstacle = ray_trace_to_obstacle(head, game.snake.direction, game.obstacles)

        # Normalize the distance to obstacle for consistency with other state features
        # normalized_distance_to_obstacle = normalize_distance(distance_to_obstacle, game.max_possible_distance)

        state = np.array([
            *snake_vision,
            food_left, food_right, food_above, food_below,
            moving_up, moving_down, moving_left, moving_right,
            # distance_to_wall_left, distance_to_wall_right, distance_to_wall_up, distance_to_wall_down,
            ])

        state = torch.from_numpy(np.array(state, dtype=float)).to(self.device)
        return state

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) < BATCH_SIZE:
            return  # Not enough samples in the memory yet

        transitions = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state in transitions:
           self.trainer.train_step(state, action, reward, next_state)

    def train_short_memory(self, state, action, reward, next_state):
        self.trainer.train_step(state, action, reward, next_state)

    def get_action(self, state, is_train=True):
        # Linear decay rate
        if is_train:
            if self.n_games > EPSILON_SHIFT:
                self.epsilon = (self.epochs - self.n_games) / (self.epochs - EPSILON_SHIFT)
        else:
            self.epsilon = SNAKE_MIN_EPSILON

        self.epsilon = max(self.epsilon, SNAKE_MIN_EPSILON)

        final_move = [0] * AVAILABLE_SNAKE_DIRECTIONS_QUANTITY
        if np.random.rand() < self.epsilon:
            move = random.randint(0, AVAILABLE_SNAKE_DIRECTIONS_QUANTITY - 1)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move
