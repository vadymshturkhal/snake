import torch
import random
import numpy as np
from collections import deque
from model import Linear_QNet, QTrainer

from game_utils import Direction, Point
from game_settings import EPSILON_SHIFT, MAX_MEMORY, BATCH_SIZE, LR, SNAKE_ACTION_LENGTH, BLOCK_SIZE, REWARD_CRAWLING, REWARD_LOOSE, REWARD_ROTATION, REWARD_WIN, SNAKE_GAMMA, SNAKE_MIN_EPSILON
from game_settings import SNAKE_INPUT_LAYER_SIZE, SNAKE_HIDDEN_LAYER_SIZE1, SNAKE_HIDDEN_LAYER_SIZE2, SNAKE_OUTPUT_LAYER_SIZE
from game_settings import CODE_SNAKE, CODE_OBSTACLES, CODE_FOOD

class SnakeAgent:
    def __init__(self, is_load_weights=False, weights_filename=None, epochs=100, is_load_n_games=True, vision_range=1):
        self.epsilon = 1
        self.epochs = epochs
        self.vision_range = vision_range

        self.gamma = SNAKE_GAMMA
        self.memory = deque(maxlen=MAX_MEMORY)

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = Linear_QNet(SNAKE_INPUT_LAYER_SIZE, SNAKE_HIDDEN_LAYER_SIZE1, SNAKE_HIDDEN_LAYER_SIZE2, SNAKE_OUTPUT_LAYER_SIZE)
        self.model.to(self.device)
        self.last_action = [0] * SNAKE_ACTION_LENGTH
        self.last_reward = 0

        # Load the weights onto the CPU or GPU
        if is_load_weights:
            checkpoint = torch.load(weights_filename, map_location=self.device)
            self.n_games = checkpoint['epoch'] if is_load_n_games else 0
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
        else:
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
        self.populate_grid(game, state_grid, center_x, center_y, vision_range)
        
        relative_state_grid = self.rotate_grid(state_grid, game.snake.direction)

        # Flatten the grid to create a state vector or use as is for CNN input
        relative_state_vector = relative_state_grid.flatten()

        return relative_state_vector

    def populate_grid(self, game, state_grid, center_x, center_y, max_vision_range):
        """
        Populate the vision grid layer by layer, from outermost to innermost.
        """
        # Iterate from the outermost layer inwards
        for vision_range in range(max_vision_range, 0, -1):
            for i in range(-vision_range, vision_range + 1):
                for j in range(-vision_range, vision_range + 1):
                    # Calculate the position for the current cell
                    point_x = center_x + i * BLOCK_SIZE
                    point_y = center_y + j * BLOCK_SIZE
                    point = Point(point_x, point_y)

                    # Mapping the calculated point to the grid's indexing system
                    grid_x = j + max_vision_range
                    grid_y = i + max_vision_range

                    # Check and assign values based on game state, similar to your existing logic
                    if point.x < 0 or point.y < 0 or point.x >= game.width or point.y >= game.height:
                        # Wall
                        state_grid[grid_x, grid_y] = REWARD_LOOSE
                    elif point in game.foods:
                         # Food
                        state_grid[grid_x, grid_y] = REWARD_WIN 
                    elif game.obstacles.is_point_at_obstacle(point):
                         # Obstacle
                        state_grid[grid_x, grid_y] = REWARD_LOOSE
                    else:
                        state_grid[grid_x, grid_y] = REWARD_CRAWLING

        state_grid[max_vision_range, max_vision_range] = REWARD_ROTATION
        # Note: This approach does not explicitly clear each layer before populating,
        # as each layer overrides its values based on current game state.

    def rotate_grid(self, grid, direction):
        """
        Rotate the grid so that it aligns with the snake's current direction.

        Parameters:
        - grid: The vision grid as a numpy array.
        - direction: The current direction of the snake.

        Returns:
        - Rotated grid as a numpy array.
        """
        if direction == Direction.UP:
            # No rotation needed.
            return grid
        elif direction == Direction.LEFT:
            # Rotate -90 degrees.
            return np.rot90(grid, -1)
        elif direction == Direction.RIGHT:
            # Rotate 90 degrees
            return np.rot90(grid, 1)
        elif direction == Direction.DOWN:
            # Rotate 180 degrees.
            return np.rot90(grid, 2)

    def get_state(self, game):
        head = game.snake.head

        snake_vision = self.get_vision_based_state(game, vision_range=self.vision_range)

        closest_food = game.foods.get_closest_food(head)

        if closest_food is None:
            food_left = False
            food_right = False
            food_above = False
            food_below = False
        else:
            # Relative food location based on snake's current direction
            if game.snake.direction == Direction.UP:
                food_left = closest_food.x < head.x
                food_right = closest_food.x > head.x
                food_above = closest_food.y < head.y
                food_below = closest_food.y > head.y
            elif game.snake.direction == Direction.DOWN:
                food_left = closest_food.x > head.x
                food_right = closest_food.x < head.x
                food_above = closest_food.y > head.y
                food_below = closest_food.y < head.y
            elif game.snake.direction == Direction.LEFT:
                food_left = closest_food.y > head.y
                food_right = closest_food.y < head.y
                food_above = closest_food.x < head.x
                food_below = closest_food.x > head.x
            elif game.snake.direction == Direction.RIGHT:
                food_left = closest_food.y < head.y
                food_right = closest_food.y > head.y
                food_above = closest_food.x > head.x
                food_below = closest_food.x < head.x

        moving_left = game.snake.direction == Direction.LEFT
        moving_right = game.snake.direction == Direction.RIGHT
        moving_up = game.snake.direction == Direction.UP
        moving_down = game.snake.direction == Direction.DOWN

        state = np.array([
            *self.last_action,
            *game.snake.head,
            self.last_reward,
            moving_up, moving_down, moving_left, moving_right,
            food_left, food_right, food_above, food_below,
            *snake_vision,
            ])

        state = torch.from_numpy(np.array(state, dtype=float)).to(self.device)
        return state

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) < BATCH_SIZE:
            transitions = self.memory
        else:
            transitions = random.sample(self.memory, BATCH_SIZE)

        for state, action, reward, next_state, done in transitions:
           self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state, is_train=True):
        """
        Return vector [0,0,0] with first two positions represent rotation, third pushes forward.
        Adjusted for epsilon-soft policy.
        """
        # Linear decay rate
        if is_train:
            if self.n_games > EPSILON_SHIFT:
                self.epsilon = (self.epochs - self.n_games) / (self.epochs - EPSILON_SHIFT)
        else:
            self.epsilon = SNAKE_MIN_EPSILON

        self.epsilon = max(self.epsilon, SNAKE_MIN_EPSILON)

        final_move = [0] * SNAKE_ACTION_LENGTH

        if np.random.rand() < self.epsilon:
            # Distribute a small probability to all actions, keeping the majority for the best action
            probabilities = np.ones(SNAKE_ACTION_LENGTH) * (self.epsilon / SNAKE_ACTION_LENGTH)

            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            best_move = torch.argmax(prediction).item()

            # Adjust probability for the best action
            probabilities[best_move] += (1.0 - self.epsilon)

            # Choose action based on modified probabilities
            move = np.random.choice(np.arange(SNAKE_ACTION_LENGTH), p=probabilities)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        self.last_action = final_move
        return final_move
