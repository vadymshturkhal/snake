from functools import reduce
import operator
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from game_settings import SNAKE_ACTION_LENGTH, WEIGHT_DECAY


class NStepOffPolicyQSigmaTrainer:
    def __init__(self, model, lr, gamma, n_steps=0):
        self._alpha = lr
        self._gamma = gamma
        self._model = model
        self._optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
        self._criterion = nn.SmoothL1Loss()
        self._n_steps = n_steps

    def train_n_steps(self, states: list, actions: list, rewards: list, dones: int, last_index=0, epsilon=1):
        """Rewards and dones 1 greater then others"""
        if len(states) < self._n_steps:
            return 0

        if last_index == 0:
            last_index = len(states)

        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long)

        if last_index is None:
            last_index = len(states)
        start_index = last_index - self._n_steps

        for t in reversed(range(start_index + 1, last_index)):
            G = 0
            if dones[t]:
                G = rewards[t]
            else:
                Q_values = self._model(states[t])
                pi, b = self._get_pi_and_b(states[t], epsilon)
                rho = pi / b

                # V = Expected Q value for all actions in the state
                V = torch.sum(F.softmax(Q_values, dim=-1) * Q_values)

                # G update
                sigma = 0 if t % 2 == 0 else 1

                # Computing the TD target for the current state and action
                current_Q = Q_values.gather(0, actions[t])  # Assuming actions[t] is the index of the action taken at time t
                G = rewards[t] + self._gamma * (sigma * rho + (1 - sigma) * pi) * (G - torch.max(current_Q).detach()) + self._gamma * V


        # Update from first to last
        q_values = self._model(states[start_index])

        target = q_values.clone()
        target[torch.argmax(actions[start_index]).item()] = G
    
        self._optimizer.zero_grad()
        loss = self._criterion(target, q_values)
        loss.backward()

        self._optimizer.step()
        return loss.item()

    def train_episode(self):
        pass

    def _calculate_rewards(self, rewards, last_index=None):
        rewards_gamma_sum = 0
        if last_index is None:
            last_index = len(rewards)
        start_index = last_index - self._n_steps

        for i in range(start_index, last_index):
            rewards_gamma_sum += rewards[i] * self._gamma**(i - start_index)
        return rewards_gamma_sum

    def _get_pi_and_b(self, state, epsilon):
        with torch.no_grad():
            logits = self._model(state)

            # Apply softmax to convert logits to probabilities
            probabilities = F.softmax(logits, dim=-1)
            action_index = torch.argmax(probabilities)

            # B
            # b = epsilon * (1 / SNAKE_ACTION_LENGTH) + (1 - epsilon)
            # Assuming the action is chosen using epsilon-greedy policy
            # First, initialize all probabilities to epsilon / number of actions
            b = torch.full(probabilities.shape, epsilon / SNAKE_ACTION_LENGTH)
            # The index of the best action (highest probability after softmax)
            best_action = torch.argmax(probabilities)

            # Now update the probability of the best action
            pi = probabilities
            b[best_action] += (1 - epsilon)

            return pi[action_index], b[best_action]
