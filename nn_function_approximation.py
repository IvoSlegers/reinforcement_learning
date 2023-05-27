import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from function_approximation import ActionValueEstimator

class Model(nn.Module):
    def __init__(self, n_in_features: int, n_out_features: int) -> None:
        super().__init__()

        self.dense1 = nn.Linear(n_in_features, 128)
        self.dense2 = nn.Linear(128, 128)
        self.dense3 = nn.Linear(128, n_out_features, bias=False)

        self.dense3.weight.detach().zero_()

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        return self.dense3(x)

class NNEstimator(ActionValueEstimator):
    def __init__(self, n_features: int, n_actions: int) -> None:
        super().__init__()

        self.model = Model(n_features, n_actions)

        self.optimizer = torch.optim.SGD(self.model.parameters(), 0.1, 0.95)

    def action_values(self, observation) -> np.array:
        with torch.no_grad():
            values = self.model(torch.from_numpy(observation))
            return values.numpy()

    def update(self, observation: int, action: int, target: float, alpha: float):
        for group in self.optimizer.param_groups:
            group['lr'] = alpha

        self.optimizer.zero_grad()

        x = torch.from_numpy(observation)
        values = self.model(x)
        action_value = values[action]

        loss = 0.5 * torch.square(action_value - target)
        loss.backward()

        self.optimizer.step()