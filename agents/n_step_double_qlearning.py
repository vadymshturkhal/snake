import torch
import numpy as np
from model import Linear_QNet
from trainers.n_step_double_qtrainer import NStepDoubleQTrainer

from game_settings import EPSILON_SHIFT, LR, SNAKE_ACTION_LENGTH, BLOCK_SIZE, TRAINER_STEPS
from game_settings import SNAKE_INPUT_LAYER_SIZE, SNAKE_HIDDEN_LAYER_SIZE1, SNAKE_HIDDEN_LAYER_SIZE2, SNAKE_OUTPUT_LAYER_SIZE
from game_settings import SNAKE_GAMMA, SNAKE_MIN_EPSILON, SNAKE_START_EPSILON


class NStepDoubleQLearning:
    def __init__(self, is_load_weights=False, weights_filename=None, epochs=100, is_load_n_games=True, n_steps=TRAINER_STEPS):
        self.epsilon = 1
        self.epochs = epochs

        self.gamma = SNAKE_GAMMA

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model1 = Linear_QNet(SNAKE_INPUT_LAYER_SIZE, SNAKE_HIDDEN_LAYER_SIZE1, SNAKE_HIDDEN_LAYER_SIZE2, SNAKE_OUTPUT_LAYER_SIZE)
        self.model2 = Linear_QNet(SNAKE_INPUT_LAYER_SIZE, SNAKE_HIDDEN_LAYER_SIZE1, SNAKE_HIDDEN_LAYER_SIZE2, SNAKE_OUTPUT_LAYER_SIZE)
        self.model1.to(self.device)
        self.model2.to(self.device)
        self.last_action = [0] * SNAKE_ACTION_LENGTH

        # Load the weights onto the CPU or GPU
        if is_load_weights:
            checkpoint = torch.load(weights_filename, map_location=self.device)
            self.n_games = checkpoint['epoch'] if is_load_n_games else 0
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
        else:
            self.n_games = 0

        self.trainer = NStepDoubleQTrainer(self.model1, self.model2, lr=LR, gamma=self.gamma, n_steps=n_steps)

    @property
    def model(self):
        return self.model1

    def train_n_steps(self, states: list, actions: list, rewards: list, dones: list) -> float:
        loss = self.trainer.train_n_steps(states, actions, rewards, dones)
        return loss

    def get_action(self, state, is_train=True):
        """
        Return vector [0,0,0] with first two positions represent rotation, third pushes forward.
        Adjusted for epsilon-soft policy.
        """
        # Linear decay rate
        if is_train:
            self._update_epsilon_linear()
        else:
            self.epsilon = SNAKE_MIN_EPSILON

        final_move = [0] * SNAKE_ACTION_LENGTH

        if np.random.rand() < self.epsilon:
            # Distribute a small probability to all actions, keeping the majority for the best action
            probabilities = np.ones(SNAKE_ACTION_LENGTH) * (self.epsilon / SNAKE_ACTION_LENGTH)

            state_tensor = torch.tensor(state, dtype=torch.float)
            with torch.no_grad():  # No need to track gradients here
                prediction1 = self.model1(state_tensor)
                prediction2 = self.model2(state_tensor)
                average_prediction = (prediction1 + prediction2) / 2
                action = torch.argmax(average_prediction).item()

            # Adjust probability for the best action
            probabilities[action] += (1.0 - self.epsilon)

            # Choose action based on modified probabilities
            move = np.random.choice(np.arange(SNAKE_ACTION_LENGTH), p=probabilities)
            final_move[move] = 1
        else:
            state_tensor = torch.tensor(state, dtype=torch.float)

            with torch.no_grad():  # No need to track gradients here
                prediction1 = self.model1(state_tensor)
                prediction2 = self.model2(state_tensor)
                average_prediction = (prediction1 + prediction2) / 2
                action = torch.argmax(average_prediction).item()

            final_move[action] = 1

        self.last_action = final_move
        return final_move

    def _update_epsilon_linear(self):
        """
        Update the epsilon value linearly from start_epsilon to end_epsilon over total_games.
        
        Parameters:
        - current_game: The current game number (1 to total_games).
        - start_epsilon: The starting value of epsilon at game 1.
        - end_epsilon: The final value of epsilon at game total_games.
        - total_games: The total number of games over which epsilon will decay.
        
        Returns:
        - Updated epsilon value for the current game.
        """

        # Calculate the amount of decay per game
        decay_per_game = (SNAKE_START_EPSILON - SNAKE_MIN_EPSILON) / (self.epochs)
        
        # Update epsilon linearly based on the current game number
        new_epsilon = SNAKE_START_EPSILON - (decay_per_game * (self.n_games))
        
        # Ensure epsilon does not go below the end_epsilon
        self.epsilon = new_epsilon
        self.epsilon = max(self.epsilon, SNAKE_MIN_EPSILON)
