import torch
import random
import numpy as np
from collections import deque
from model import Linear_QNet, QTrainer

from game_utils import Direction, Point, calculate_distance, check_dangers, normalize_distance, calculate_angle, ray_trace_to_obstacle
from game_settings import MAX_MEMORY, BATCH_SIZE, LR, AVAILABLE_SNAKE_DIRECTIONS_QUANTITY, BLOCK_SIZE
from game_settings import SNAKE_INPUT_LAYER_SIZE, SNAKE_HIDDEN_LAYER_SIZE1, SNAKE_HIDDEN_LAYER_SIZE2, SNAKE_OUTPUT_LAYER_SIZE


class SnakeAgent:
    def __init__(self, is_load_weights=False, weights_filename=None, epochs=100):
        self.epsilon = epochs # Starting value of epsilon
        self.min_epsilon = 0.001
        self.epochs = epochs

        self.gamma = 0.9 # discount rate
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

    def get_state(self, game):
        head = game.snake.head

        # dangers = check_dangers(game)

        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        point_ul = Point(head.x - BLOCK_SIZE, head.y - BLOCK_SIZE)
        point_ur = Point(head.x + BLOCK_SIZE, head.y - BLOCK_SIZE)
        point_dl = Point(head.x - BLOCK_SIZE, head.y + BLOCK_SIZE)
        point_dr = Point(head.x + BLOCK_SIZE, head.y + BLOCK_SIZE)

        dir_l = game.snake.direction == Direction.LEFT
        dir_r = game.snake.direction == Direction.RIGHT
        dir_u = game.snake.direction == Direction.UP
        dir_d = game.snake.direction == Direction.DOWN

        # Assuming snake_head and food_position are Point objects with x and y attributes
        distance = calculate_distance(head, game.food.head)
        angle = calculate_angle(game.snake, game.food.head)

        # Assuming `state` is your current state vector and `current_angle` has been calculated
        normalized_angle = angle / 360  # Example normalization if angle is in degrees

        # Example usage
        normalized_distance = normalize_distance(distance, game.max_possible_distance)

        # Obstacles
        # closest_point, distance_to_closest_obstacle = sort_obstacles(head, game.obstacles)[0]

        # Additional danger checks for diagonal movements
        danger_ur = (dir_u and game.is_collision(point_ur)) or (dir_r and game.is_collision(point_ur))
        danger_ul = (dir_u and game.is_collision(point_ul)) or (dir_l and game.is_collision(point_ul))
        danger_dr = (dir_d and game.is_collision(point_dr)) or (dir_r and game.is_collision(point_dr))
        danger_dl = (dir_d and game.is_collision(point_dl)) or (dir_l and game.is_collision(point_dl))

         # Add ray tracing distance to the state
        distance_to_obstacle = ray_trace_to_obstacle(head, game.snake.direction, game.obstacles)

        # Normalize the distance to obstacle for consistency with other state features
        normalized_distance_to_obstacle = normalize_distance(distance_to_obstacle, game.max_possible_distance)

        state = np.array([
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Danger Behind
            (dir_d and game.is_collision(point_u)) or
            (dir_u and game.is_collision(point_d)) or
            (dir_r and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_r)),

            danger_ur,
            danger_ul,
            danger_dr,
            danger_dl,

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.head.x < head.x,  # food left
            game.food.head.x > head.x,  # food right
            game.food.head.y < head.y,  # food up
            game.food.head.y > head.y,  # food down

            normalized_distance,
            normalized_angle,
            normalized_distance_to_obstacle
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

    def get_action(self, state):
        # Linear decay rate
        # random moves: tradeoff exploration / exploitation
        self.epsilon = (self.epochs - self.n_games) / self.epochs

        final_move = [0] * AVAILABLE_SNAKE_DIRECTIONS_QUANTITY
        if np.random.rand() < max(self.epsilon, self.min_epsilon):
            move = random.randint(0, AVAILABLE_SNAKE_DIRECTIONS_QUANTITY - 1)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move
