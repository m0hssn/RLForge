import torch

class RLAgent:
    def __init__(self, policy_learner, replay_buffer):
        self.policy_learner = policy_learner
        self.replay_buffer = replay_buffer

    def reset(self, observation, action_space):
        self.observation = observation
        self.action_space = action_space
        self.policy_learner.reset()

    def act(self, exploit=False):
        return self.policy_learner.select_action(torch.tensor(self.observation), exploit)

    def observe(self, action_result):
        self.replay_buffer.store((self.observation, action_result.action, action_result.reward, action_result.next_state, action_result.done))
        self.observation = action_result.next_state

    def learn(self):
        if len(self.replay_buffer) > self.policy_learner.batch_size:
            batch = self.replay_buffer.sample(self.policy_learner.batch_size)
            self.policy_learner.update_policy(batch)
