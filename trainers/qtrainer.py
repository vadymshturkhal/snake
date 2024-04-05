import torch
import torch.nn as nn
import torch.optim as optim

from game_settings import WEIGHT_DECAY


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=WEIGHT_DECAY)
        # self.criterion = nn.MSELoss()
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

        # Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        if not done:
            q_update = reward[0] + self.gamma * torch.max(self.model(next_state).detach())
        else:
            q_update = reward[0]

        q_values = self.model(state)

        target = q_values.clone()
        target[0][torch.argmax(action[0]).item()] = q_update
    
        self.optimizer.zero_grad()
        loss = self.criterion(target, q_values)
        loss.backward()

        self.optimizer.step()
        return loss.item()
