import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from game_settings import DROPOUT_RATE, WEIGHT_DECAY


class Linear_QNet(nn.Module):
    def __init__(self, input_layer, hidden1, hidden2, output_layer):
        super().__init__()
        self.linear1 = nn.Linear(input_layer, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.dropout1 = nn.Dropout(DROPOUT_RATE)
        self.linear2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.dropout2 = nn.Dropout(DROPOUT_RATE)
        self.linear3 = nn.Linear(hidden2, output_layer)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout1(x)
        x = F.relu(self.linear2(x))
        x = self.dropout2(x)
        x = self.linear3(x)
        return x

    def save(self, epoch=0, filename=None):
        torch.save({
            'model_state_dict': self.state_dict(),
            'epoch': epoch,
            }, filename)

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
            model_to_update = self.model1
            model_to_estimate = self.model2
            optimizer = self.optimizer1
        else:
            model_to_update = self.model2
            model_to_estimate = self.model1
            optimizer = self.optimizer2

        # Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        if not done:
            # q_update = reward[0] + self.gamma * torch.max(self.model(next_state).detach())
            q_update = reward[0] + self.gamma * model_to_estimate(next_state).detach()
        else:
            q_update = reward[0]

        q_values = model_to_update(state)

        target = q_values.clone()
        target[0][torch.argmax(action[0]).item()] = q_update
    
        optimizer.zero_grad()
        loss = self.criterion(target, q_values)
        loss.backward()

        optimizer.step()
        return loss.item()
