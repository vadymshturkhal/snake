import random
import torch
import torch.nn as nn
import torch.optim as optim

from game_settings import WEIGHT_DECAY


class NStepDoubleQTrainer:
    def __init__(self, model1, model2, lr, gamma, n_steps=1):
        self._gamma = gamma
        self._model1 = model1
        self._model2 = model2
        self._optimizer1 = optim.Adam(model1.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
        self._optimizer2 = optim.Adam(model2.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
        self._criterion = nn.SmoothL1Loss()
        self._n_steps = n_steps

    def train_n_steps(self, states: list, actions: list, rewards: list, dones: int):
        if len(states) < self._n_steps:
            return 0

        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long)

        if random.random() > 0.5:
            model1 = self._model1
            model2 = self._model2
            optimizer = self._optimizer1
        else:
            model1 = self._model2
            model2 = self._model1
            optimizer = self._optimizer2
        
        # G
        rewards_gamma_sum = self.calculate_rewards(rewards)

        if not dones[-1]:
            # G + y**n * max(Q(S_tau+n, a_tau+n))
            rewards_gamma_sum += self._gamma**self._n_steps * torch.max(model2(states[-1]).detach())

        q_values = model1(states[-self._n_steps])

        target = q_values.clone()
        target[torch.argmax(actions[-self._n_steps]).item()] = rewards_gamma_sum
    
        optimizer.zero_grad()
        loss = self._criterion(target, q_values)
        loss.backward()

        optimizer.step()
        return loss.item()

    def _get_random_models(self):
        # Randomly choose which Q-network to update
        if random.random() > 0.5:
            model1 = self._model1
            model2 = self._model2
            optimizer = self._optimizer1
        else:
            model1 = self._model2
            model2 = self._model1
            optimizer = self._optimizer2
        return model1, model2, optimizer

    def train_episode(self):
        pass

    def calculate_rewards(self, rewards):
        rewards_gamma_sum = 0
        start_index = len(rewards) - self._n_steps

        for i in range(start_index, len(rewards)):
            rewards_gamma_sum += rewards[i] * self._gamma**(i - start_index)
        return rewards_gamma_sum
