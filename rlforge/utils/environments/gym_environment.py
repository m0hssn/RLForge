import gymnasium as gym
import torch

class GymEnvironment:
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        state, _ = self.env.reset()
        return torch.tensor(state, dtype=torch.float32)

    def step(self, action):
        next_state, reward, done, _, _ = self.env.step(action)
      
        return ActionResult(
            next_state=torch.tensor(next_state, dtype=torch.float32),
            reward=torch.tensor(reward, dtype=torch.float32),
            done=torch.tensor(done, dtype=torch.float32),
            action=torch.tensor(action, dtype=torch.int64)
        )

class ActionResult:
    def __init__(self, next_state, reward, done, action):
        self.next_state = next_state
        self.reward = reward
        self.done = done
        self.action = action
