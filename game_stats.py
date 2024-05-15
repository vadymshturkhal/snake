import numpy as np
import torch
from game_settings import BLOCK_SIZE, REWARD_CRAWLING, REWARD_LOOSE, REWARD_ROTATION, REWARD_WIN, SNAKE_VISION_RANGE
import pprint

from game_utils import Direction, Point, calculate_distance, rotate_grid


class GameStats:
    def __init__(self, game):
        self.game = game
        self._vision_range = SNAKE_VISION_RANGE
        self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def get_snake_state(self) -> np.array:
        head = self.game.snake.head

        snake_vision = self._get_area_around()

        closest_food = self.game.foods.get_closest_food(head)

        if closest_food is None:
            food_left = False
            food_right = False
            food_above = False
            food_below = False
        else:
            # Relative food location based on snake's current direction
            if self.game.snake.direction == Direction.UP:
                food_left = closest_food.x < head.x
                food_right = closest_food.x > head.x
                food_above = closest_food.y < head.y
                food_below = closest_food.y > head.y
            elif self.game.snake.direction == Direction.DOWN:
                food_left = closest_food.x > head.x
                food_right = closest_food.x < head.x
                food_above = closest_food.y > head.y
                food_below = closest_food.y < head.y
            elif self.game.snake.direction == Direction.LEFT:
                food_left = closest_food.y > head.y
                food_right = closest_food.y < head.y
                food_above = closest_food.x < head.x
                food_below = closest_food.x > head.x
            elif self.game.snake.direction == Direction.RIGHT:
                food_left = closest_food.y < head.y
                food_right = closest_food.y > head.y
                food_above = closest_food.x > head.x
                food_below = closest_food.x < head.x

        moving_left = self.game.snake.direction == Direction.LEFT
        moving_right = self.game.snake.direction == Direction.RIGHT
        moving_up = self.game.snake.direction == Direction.UP
        moving_down = self.game.snake.direction == Direction.DOWN

        closest_food = self.game.foods.get_closest_food(self.game.snake.head)
        if closest_food is None:
            distance_between_snake_food = 0
        else:
            distance_between_snake_food = calculate_distance(self.game.snake.head, self.game.foods.get_closest_food(self.game.snake.head))

        state = np.array([
            moving_up, moving_down, moving_left, moving_right,
            food_left, food_right, food_above, food_below,
            *snake_vision,
            ])

        # state = torch.from_numpy(np.array(state, dtype=float)).to(self._device)
        return state

    def _get_area_around(self):
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
        grid_size = 2 * self._vision_range + 1  # Calculate the size of the vision grid
        state_grid = np.zeros((grid_size, grid_size))  # Initialize the grid to zeros

        # Define the grid's center point (the snake's head position)
        center_x, center_y = self.game.snake.head.x, self.game.snake.head.y
        self._populate_vision_around_grid(self.game, state_grid, center_x, center_y, self._vision_range)
        
        relative_state_grid = rotate_grid(state_grid, self.game.snake.direction)

        # Flatten the grid to create a state vector or use as is for CNN input
        relative_state_vector = relative_state_grid.flatten()

        print(relative_state_grid)

        return relative_state_vector

    def _populate_vision_around_grid(self, game, state_grid, center_x, center_y, max_vision_range):
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
                        state_grid[grid_x, grid_y] = 1
                    elif point in game.foods:
                         # Food
                        state_grid[grid_x, grid_y] = 2
                    elif game.obstacles.is_point_at_obstacle(point):
                         # Obstacle
                        state_grid[grid_x, grid_y] = 1
                    else:
                        state_grid[grid_x, grid_y] = 0

        state_grid[max_vision_range, max_vision_range] = REWARD_ROTATION
