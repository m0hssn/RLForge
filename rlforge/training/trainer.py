import torch

class Trainer:
    def __init__(self, env, agent, max_episodes=1000, max_steps=200):
        self.env = env
        self.agent = agent
        self.max_episodes = max_episodes
        self.max_steps = max_steps

    def train(self):
        for episode in range(self.max_episodes):
            observation, _ = self.env.reset()
            self.agent.reset(observation, self.env.action_space)
            done = False
            total_reward = 0
            
            for step in range(self.max_steps):
                action = self.agent.act(exploit=False)  # Sample action
                action_result = self.env.step(action)
                
                self.agent.observe(action_result)  # Observe the result
                self.agent.learn()  # Learn from the observed result
                
                total_reward += action_result.reward  # Accumulate rewards
                done = action_result.done
                
                if done:
                    break
            
            print(f"Episode {episode + 1}/{self.max_episodes} - Total Reward: {total_reward}")

        print("Training complete.")
