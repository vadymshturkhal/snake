import torch
import random
import numpy as np
from collections import deque
from game_utils import Direction, Point, DEVICE, calculate_distance, normalize_distance, calculate_angle
from model import Linear_QNet, QTrainer
from game_settings import MAX_MEMORY, BATCH_SIZE, LR, AVAILABLE_SNAKE_DIRECTIONS_QUANTITY
from game_settings import BLOCK_SIZE
from game_settings import FOOD_INPUT_LAYER_SIZE, FOOD_HIDDEN_LAYER_SIZE1, FOOD_HIDDEN_LAYER_SIZE2, FOOD_OUTPUT_LAYER_SIZE


# Add a reward for any safe step
class FoodAgent:
    def __init__(self, is_load_weights=False, weights_filename=None, epochs=100):
        self.epsilon = 100 # Starting value of epsilon
        self.epochs = epochs

        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = Linear_QNet(FOOD_INPUT_LAYER_SIZE, FOOD_HIDDEN_LAYER_SIZE1, FOOD_HIDDEN_LAYER_SIZE2, FOOD_OUTPUT_LAYER_SIZE)
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
        food = game.food
        point_l = Point(food.head.x - BLOCK_SIZE, food.head.y)
        point_r = Point(food.head.x + BLOCK_SIZE, food.head.y)
        point_u = Point(food.head.x, food.head.y - BLOCK_SIZE)
        point_d = Point(food.head.x, food.head.y + BLOCK_SIZE)

        dir_l = game.snake.direction == Direction.LEFT
        dir_r = game.snake.direction == Direction.RIGHT
        dir_u = game.snake.direction == Direction.UP
        dir_d = game.snake.direction == Direction.DOWN

        # Assuming snake_head and food_position are Point objects with x and y attributes
        distance = calculate_distance(food.head, game.snake.head)
        angle = calculate_angle(food.head, game.snake.head)

        # Example usage
        normalized_distance = normalize_distance(distance, game.max_possible_distance)

        state = [
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

            # Danger behind
            (dir_r and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_d)) or
            (dir_d and game.is_collision(point_u)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Snake location
            food.head.x < game.snake.head.x, 
            food.head.x > game.snake.head.x, 
            food.head.y < game.snake.head.y,  
            food.head.y > game.snake.head.y,

            normalized_distance,
            angle,
            ]

        state = torch.from_numpy(np.array(state, dtype=int)).to(self.device)
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
        self.epsilon = self.epochs - self.n_games

        final_move = [0] * 4
        if random.randint(0, self.epochs) < self.epsilon:
            move = random.randint(0, AVAILABLE_SNAKE_DIRECTIONS_QUANTITY - 1)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move
