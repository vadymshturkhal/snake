from functools import reduce
import operator
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from game_settings import SNAKE_ACTION_LENGTH, WEIGHT_DECAY, MIN_LR, MAX_LR


class StochasticTrainer:
    def __init__(self, model, lr, gamma):
        self._alpha = lr
        self._gamma = gamma
        self._model = model
        self._optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY, momentum=0.9, foreach=False)
        self._criterion = nn.MSELoss()

    def train_episode(self, states: list, actions: list, rewards: list, dones: int):
        if len(states) < 2:
            return 0

        losses = []
        for i in range(2, len(states)):
            last_index = i
            loss = self._train_episode(states, actions, rewards, dones, last_index)
            losses.append(loss)

        return sum(losses) / len(losses)

    def _train_episode(self, states: list, actions: list, rewards: list, dones: int, last_index=0):
        states = torch.tensor(np.array(states), dtype=torch.float)
        actions = torch.tensor(np.array(actions), dtype=torch.long)

        # Perform a forward pass to get the value estimate
        state_tensor = torch.tensor(states[last_index], dtype=torch.float32)
        value_estimate = self._model(state_tensor)

        # G
        rewards_gamma_sum = self._calculate_rewards(rewards, last_index=last_index)

        # Calculate the loss as the squared difference (ensure it is a scalar)
        loss = (rewards_gamma_sum - value_estimate).pow(2).mean()  # .mean() to ensure the loss is a scalar

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        return loss.item()

    def _calculate_rewards(self, rewards, last_index=None):
        rewards_gamma_sum = 0
        if last_index is None:
            last_index = len(rewards)

        for i in range(last_index + 1):
            rewards_gamma_sum += rewards[i] * self._gamma**(i)
        return rewards_gamma_sum
