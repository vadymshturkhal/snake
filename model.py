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

    def train_step(self, state, action, reward, next_state):
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

        # 1: predicted Q values with current state
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        q_update = reward[0] + self.gamma * torch.max(self.model(next_state).detach())
        q_values = self.model(state)

        target = q_values.clone()
        target[0][torch.argmax(action[0]).item()] = q_update
    
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, q_values)
        loss.backward()

        self.optimizer.step()
