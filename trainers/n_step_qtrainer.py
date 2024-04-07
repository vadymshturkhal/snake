import random
import torch
import torch.nn as nn
import torch.optim as optim

from game_settings import WEIGHT_DECAY


class NStepQTrainer:
    def __init__(self, model, lr, gamma, n_steps=0):
        self._gamma = gamma
        self._model = model
        self._optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
        self._criterion = nn.SmoothL1Loss()
        self._n_steps = n_steps

    def train_n_steps(self, states: list, actions: list, rewards: list, dones: int):
        if len(states) < self._n_steps:
            return 0

        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long)

        # G
        rewards_gamma_sum = self.calculate_rewards(rewards)

        if not dones[-1]:
            # G + y**n * max(Q(S_tau+n, a_tau+n))
            rewards_gamma_sum += self._gamma**self._n_steps * torch.max(self._model(states[-1]).detach())

        q_values = self._model(states[-self._n_steps])

        target = q_values.clone()
        target[torch.argmax(actions[-self._n_steps]).item()] = rewards_gamma_sum
    
        self._optimizer.zero_grad()
        loss = self._criterion(target, q_values)
        loss.backward()

        self._optimizer.step()
        return loss.item()

    def train_episode(self):
        pass

    def calculate_rewards(self, rewards):
        rewards_gamma_sum = 0
        start_index = len(rewards) - self._n_steps
        for i in range(start_index, len(rewards)):
            rewards_gamma_sum += rewards[i] * self._gamma**(i - start_index)
        return rewards_gamma_sum
