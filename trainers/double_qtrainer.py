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

        q_values1 = model_to_update(state)
        q_values2 = model_to_estimate(state)
        if not done:

            with torch.no_grad():
                next_q_values1 = model_to_update(next_state)
                next_q_values2 = model_to_estimate(next_state)

            # Double Q-Learning update
            # Use the model2 (Q2) to select the action with the highest value in the next state
            next_actions = torch.argmax(next_q_values2, dim=1)

            # Use the selected action to get the corresponding Q value from model1 (Q1)
            next_q_values = next_q_values1[torch.arange(next_q_values1.size(0)), next_actions]
            
            # Calculate the target Q values for the current state
            q_target = reward + self.gamma * next_q_values
        else:
            q_target = reward

        # Get the Q values for the actual actions taken from model1 (Q1)
        q_expected = q_values1[torch.arange(q_values1.size(0)), action.view(-1)]

        # Calculate the loss
        loss = self.criterion(q_expected, q_target)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()
