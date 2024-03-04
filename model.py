import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Linear_QNet(nn.Module):
    def __init__(self, input_layer, hidden1, hidden2, output_layer):
        super().__init__()
        # First hidden layer
        self.linear1 = nn.Linear(input_layer, hidden1)
        # New second hidden layer, size 64
        self.linear2 = nn.Linear(hidden1, hidden2)
        # Adjusted output layer, now taking input from the new hidden layer
        self.linear3 = nn.Linear(hidden2, output_layer)

    def forward(self, x):
        # Activation for first hidden layer
        x = F.relu(self.linear1(x))
        # Activation for new second hidden layer
        x = F.relu(self.linear2(x))
        # Output layer does not usually have activation in regression tasks,
        # or might have a softmax for classification which is not shown here
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
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

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
        pred = self.model(state)

        target = pred.clone()
        Q_new = reward[0]
        Q_new = reward[0] + self.gamma * torch.max(self.model(next_state[0]))

        target[0][torch.argmax(action[0]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
