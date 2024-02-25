import torch
import random
import numpy as np
from collections import deque
from game_utils import Direction, Point
from model import Linear_QNet, QTrainer
from game_settings import MAX_MEMORY, BATCH_SIZE, LR, AVAILABLE_SNAKE_DIRECTIONS_QUANTITY
from game_settings import INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE


class SnakeAgent:
    def __init__(self, is_load_weights=False, weights_filename=None):
        self.epsilon = 1.0  # Starting value of epsilon
        self.epsilon_min = 0.01  # Minimum value of epsilon
        self.epsilon_decay = 0.095  # Decay multiplier to apply to epsilon each time

        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Load the weights onto the CPU
        if is_load_weights:
            checkpoint = torch.load(weights_filename, map_location=self.device)
            checkpoint = torch.load(weights_filename, self.device)
            
            self.n_games = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
        else:
            self.n_games = 0

        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake.head
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.snake.direction == Direction.LEFT
        dir_r = game.snake.direction == Direction.RIGHT
        dir_u = game.snake.direction == Direction.UP
        dir_d = game.snake.direction == Direction.DOWN

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

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.snake.head.x,  # food left
            game.food.x > game.snake.head.x,  # food right
            game.food.y < game.snake.head.y,  # food up
            game.food.y > game.snake.head.y  # food down
            ]
        
        state = torch.from_numpy(np.array(state, dtype=int)).to(self.device)
        return state

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        # if len(self.memory) > BATCH_SIZE:
        #     mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        # else:
        #     mini_sample = self.memory

        if len(self.memory) < BATCH_SIZE:
            return  # Not enough samples in the memory yet

        transitions = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state, done in transitions:
           self.trainer.train_step(state, action, reward, next_state, done)
           
        # states, actions, rewards, next_states, dones = zip(*mini_sample)
        # self.trainer.train_batch(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)


    def update_epsilon(self):
        """
        Applies exponential decay to epsilon.

        Args:
            epsilon (float): The current epsilon value.
            epsilon_min (float): The minimum epsilon value to ensure some level of exploration.
            epsilon_decay (float): The decay rate applied to epsilon after each episode or step.

        Returns:
            float: The updated epsilon value.
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        final_move = [0] * AVAILABLE_SNAKE_DIRECTIONS_QUANTITY
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, AVAILABLE_SNAKE_DIRECTIONS_QUANTITY - 1)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        self.update_epsilon()
        return final_move
