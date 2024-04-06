import random
import torch
import torch.nn as nn
import torch.optim as optim

from game_settings import WEIGHT_DECAY


class DoubleQTrainer:
    def __init__(self, model1, model2, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model1 = model1
        self.model2 = model2
        self.optimizer1 = optim.Adam(model1.parameters(), lr=self.lr)
        self.optimizer2 = optim.Adam(model2.parameters(), lr=self.lr)
        self.criterion = nn.SmoothL1Loss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)

        # Randomly choose which Q-network to update
        if random.random() > 0.5:
            model1 = self.model1
            model2 = self.model2
            optimizer = self.optimizer1
        else:
            model1 = self.model2
            model2 = self.model1
            optimizer = self.optimizer2

        # Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        if not done:
            q_update = reward[0] + self.gamma * torch.max(model2(next_state).detach())
        else:
            q_update = reward[0]

        q_values = model1(state)

        target = q_values.clone()
        target[0][torch.argmax(action[0]).item()] = q_update
    
        optimizer.zero_grad()
        loss = self.criterion(target, q_values)
        loss.backward()

        optimizer.step()
        return loss.item()
