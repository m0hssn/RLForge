# trainer.py
import torch

class Trainer:
    def __init__(self, env, agent, max_episodes=1000, max_steps=200, initial_epsilon=1.0, final_epsilon=0.1, epsilon_decay=0.995):
        self.env = env
        self.agent = agent
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.epsilon = initial_epsilon  
        self.final_epsilon = final_epsilon  
        self.epsilon_decay = epsilon_decay  

    def train(self):
        for episode in range(self.max_episodes):
            observation, _ = self.env.reset()
            self.agent.reset(observation, self.env.action_space)
            done = False
            total_reward = 0
            
            for step in range(self.max_steps):
                if torch.rand(1).item() < self.epsilon:
                    action = self.agent.act(exploit=False)
                else:
                    action = self.agent.act(exploit=True)
                
                action_result = self.env.step(action)
                self.agent.observe(action_result) 
                self.agent.learn() 
                
                total_reward += action_result.reward 
                done = action_result.done
                
                if done:
                    break
            
            
            self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)

            print(f"Episode {episode + 1}/{self.max_episodes} - Total Reward: {total_reward} - Epsilon: {self.epsilon:.2f}")

        print("Training complete.")
