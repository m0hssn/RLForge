import torch
from abc import ABC, abstractmethod

class ActionRepresentation(ABC):
    @abstractmethod
    def transform(self, action):
        pass

class OneHotActionRepresentation(ActionRepresentation):
    def __init__(self, num_actions):
        self.num_actions = num_actions

    def transform(self, action):
        one_hot = torch.zeros(self.num_actions)
        one_hot[action] = 1.0
        return one_hot
