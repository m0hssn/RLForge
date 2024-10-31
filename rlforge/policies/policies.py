import torch
import torch.nn as nn
import torch.optim as optim
from abc import ABC, abstractmethod

# Base Policy Learner Class
class PolicyLearner(ABC):
    def __init__(self, state_dim, action_space, action_representation, device, **kwargs):
        self.action_space = action_space
        self.action_representation = action_representation
        self.device = device

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def select_action(self, state, exploit=False):
        pass

    @abstractmethod
    def update_policy(self, batch):
        pass

# Deep Q-Learning (DQN) Policy Learner
class DeepQLearner(PolicyLearner):
    def __init__(self, state_dim, action_space, action_representation, hidden_dims, training_rounds, gamma=0.99, lr=0.001, batch_size=64, device='cpu'):
        super().__init__(state_dim, action_space, action_representation, device)
        self.gamma = gamma
        self.batch_size = batch_size
        self.q_network = self.build_network(state_dim, hidden_dims, action_space.n).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

    def build_network(self, input_dim, hidden_dims, output_dim):
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, output_dim))
        return nn.Sequential(*layers)

    def reset(self):
        pass

    def select_action(self, state, exploit=False):
        state = state.to(self.device).float()
        if exploit:
            with torch.no_grad():
                q_values = self.q_network(state)
                action = torch.argmax(q_values).item()
        else:
            action = torch.randint(0, self.action_space.n, (1,)).item()
        return action

    def update_policy(self, batch):
        states, actions, rewards, next_states, dones = map(torch.stack, zip(*batch))
        states, actions, rewards, next_states, dones = (states.to(self.device),
                                                        actions.to(self.device),
                                                        rewards.to(self.device),
                                                        next_states.to(self.device),
                                                        dones.to(self.device))
        
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        with torch.no_grad():
            next_q_values = self.q_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.functional.mse_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Double DQN Policy Learner
class DoubleDQN(DeepQLearner):
    def __init__(self, state_dim, action_space, action_representation, hidden_dims, training_rounds, gamma=0.99, lr=0.001, batch_size=64, tau=0.005, device='cpu'):
        super().__init__(state_dim, action_space, action_representation, hidden_dims, training_rounds, gamma, lr, batch_size, device)
        self.target_network = self.build_network(state_dim, hidden_dims, action_space.n).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.tau = tau

    def update_policy(self, batch):
        states, actions, rewards, next_states, dones = map(torch.stack, zip(*batch))
        states, actions, rewards, next_states, dones = (states.to(self.device),
                                                        actions.to(self.device),
                                                        rewards.to(self.device),
                                                        next_states.to(self.device),
                                                        dones.to(self.device))
        
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=1)
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.functional.mse_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

# Actor-Critic (A2C) Policy Learner
class ActorCriticPolicy(PolicyLearner):
    def __init__(self, state_dim, action_space, action_representation, hidden_dims, gamma=0.99, lr=0.001, device='cpu'):
        super().__init__(state_dim, action_space, action_representation, device)
        self.gamma = gamma
        self.actor_network = self.build_actor(state_dim, hidden_dims, action_space.n).to(self.device)
        self.critic_network = self.build_critic(state_dim, hidden_dims).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=lr)

    def build_actor(self, input_dim, hidden_dims, output_dim):
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, output_dim))
        layers.append(nn.Softmax(dim=-1))
        return nn.Sequential(*layers)

    def build_critic(self, input_dim, hidden_dims):
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        return nn.Sequential(*layers)

    def select_action(self, state, exploit=False):
        state = state.to(self.device).float()
        probs = self.actor_network(state)
        if exploit:
            action = torch.argmax(probs).item()
        else:
            action = torch.multinomial(probs, num_samples=1).item()
        return action

    def update_policy(self, batch):
        states, actions, rewards, next_states, dones = map(torch.stack, zip(*batch))
        states, actions, rewards, next_states, dones = (states.to(self.device),
                                                        actions.to(self.device),
                                                        rewards.to(self.device),
                                                        next_states.to(self.device),
                                                        dones.to(self.device))
        
        values = self.critic_network(states).squeeze()
        next_values = self.critic_network(next_states).squeeze()
        target_values = rewards + self.gamma * next_values * (1 - dones)
        critic_loss = nn.functional.mse_loss(values, target_values)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        advantages = (target_values - values).detach()
        log_probs = torch.log(self.actor_network(states).gather(1, actions.unsqueeze(1)).squeeze())
        actor_loss = -(log_probs * advantages).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

# SARSA Policy Learner
class SARSA(PolicyLearner):
    def __init__(self, state_dim, action_space, action_representation, hidden_dims, gamma=0.99, lr=0.001, epsilon=0.1, device='cpu'):
        super().__init__(state_dim, action_space, action_representation, device)
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_network = self.build_network(state_dim, hidden_dims, action_space.n).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

    def build_network(self, input_dim, hidden_dims, output_dim):
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, output_dim))
        return nn.Sequential(*layers)

    def reset(self):
        pass

    def select_action(self, state, exploit=False):
        state = state.to(self.device).float()
        if exploit or torch.rand(1).item() > self.epsilon:
            with torch.no_grad():
                q_values = self.q_network(state)
                action = torch.argmax(q_values).item()
        else:
            action = torch.randint(0, self.action_space.n, (1,)).item()
        return action

    def update_policy(self, batch):
        states, actions, rewards, next_states, next_actions, dones = map(torch.stack, zip(*batch))
        states, actions, rewards, next_states, next_actions, dones = (states.to(self.device),
                                                                      actions.to(self.device),
                                                                      rewards.to(self.device),
                                                                      next_states.to(self.device),
                                                                      next_actions.to(self.device),
                                                                      dones.to(self.device))
        
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        with torch.no_grad():
            next_q_values = self.q_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.functional.mse_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
