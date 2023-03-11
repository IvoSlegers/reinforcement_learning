import random
from abc import ABC, abstractmethod, abstractproperty
from typing import Optional


import numpy as np


class Agent(ABC):
    def __init__(self, n_actions: int) -> None:
        self.n_actions = n_actions

    @abstractmethod
    def choose_action(self, action: int) -> int:
        raise NotImplementedError()

    @abstractmethod
    def update(self, action: int, reward: float) -> None:
        raise NotImplementedError()


class PolicyAgent(Agent):
    @abstractmethod
    def get_policy(self) -> np.ndarray:
        raise NotImplementedError()

    def choose_action(self) -> int:
        policy = self.get_policy()
        return np.random.choice(np.arange(self.n_actions), p=policy).item()


class EpsilonGreedyAgent(Agent):
    def __init__(self, n_actions: int, epsilon: float, n_warmup_steps: int) -> None:
        super().__init__(n_actions)

        self.epsilon = epsilon
        self.n_warmup_steps = n_warmup_steps

        self.steps = 0
        self.total_rewards = np.zeros(shape=n_actions)

    def choose_action(self) -> int:
        if self.steps < self.n_warmup_steps or random.random() < self.epsilon:
            return random.choice(range(self.n_actions))
        else:
            return self.total_rewards.argmax().item()

    def update(self, action: int, reward: float) -> None:
        assert 0 <= action < self.n_actions
        self.steps += 1
        self.total_rewards[action] += reward


class UCBAgent(Agent):
    def __init__(self, n_actions: int) -> None:
        super().__init__(n_actions)

        self.steps = 0
        self.action_counts = np.zeros(shape=n_actions)
        self.total_rewards = np.zeros(shape=n_actions)

    def choose_action(self) -> int:
        u_bounds = np.sqrt(1/2 * np.log(self.steps + 1) / (self.action_counts + 1))
        value_estimates = self.total_rewards / (self.steps + 1)

        return (value_estimates + u_bounds).argmax().item()

    def update(self, action: int, reward: float) -> None:
        assert 0 <= action < self.n_actions
        
        self.steps += 1
        self.action_counts[action] += 1
        self.total_rewards[action] += reward


class ThompsonSamplingAgent(Agent):
    def __init__(self, n_actions: int) -> None:
        super().__init__(n_actions)

        self.alphas = np.full(shape=n_actions, fill_value=1.0)
        self.betas = np.full(shape=n_actions, fill_value=1.0)

    def choose_action(self) -> int:
        return np.random.beta(self.alphas, self.betas).argmax().item()

    def update(self, action: int, reward: float) -> None:
        
        if reward == 0:
            self.betas[action] += 1
        else:
            self.alphas[action] += 1


def softmax(x: np.ndarray) -> np.ndarray:
    exponentiated = np.exp(x)
    denumerator = exponentiated.sum()
    return exponentiated / denumerator

class PolicyOptimizationAgent(PolicyAgent):

    def __init__(self, n_action: int, learning_rate: float, baseline_ema_rate: Optional[float]) -> None:
        super().__init__(n_action)

        self.learning_rate = learning_rate
        self.baseline_ema_rate = baseline_ema_rate

        self.action_preferences = np.zeros(shape=n_action)
        self.policy = np.full(shape=n_action, fill_value=1/n_action)
        self.baseline = 0.0

    def get_policy(self) -> np.ndarray:
        return self.policy

    def update(self, action, reward):
        assert 0 <= action < self.n_actions

        if self.baseline_ema_rate is not None:
            self.baseline += (1 - self.baseline_ema_rate) * (reward - self.baseline)

        centered_reward = reward - self.baseline

        self.action_preferences -= self.learning_rate * centered_reward * self.get_policy()
        self.action_preferences[action] += self.learning_rate * centered_reward     

        self.policy = softmax(self.action_preferences)


class StatelessEnvironment(ABC):
    @abstractproperty
    def n_actions(self) -> int:
        raise NotImplementedError()

    def get_reward(self, action: int) -> float:
        raise NotImplementedError()

    def get_regret(self, action: int) -> float:
        raise NotImplementedError()

    def get_expected_regret(self, policy: np.ndarray) -> float:
        raise NotImplementedError()


class BanditEnvironment(StatelessEnvironment):
    def __init__(self, bernoulli_parameters: np.ndarray) -> None:
        super().__init__()

        self.bernoulli_parameters = bernoulli_parameters

    @property
    def n_actions(self) -> int:
        return len(self.bernoulli_parameters)

    def get_reward(self, action: int) -> float:
        assert 0 <= action < self.n_actions
        return float(random.random() < self.bernoulli_parameters[action])

    def get_regret(self, action: int) -> float:
        assert 0 <= action < self.n_actions
        return self.bernoulli_parameters.max() - self.bernoulli_parameters[action]

    def get_expected_regret(self, policy: np.ndarray) -> float:
        assert self.bernoulli_parameters.shape == policy.shape

        return np.dot(policy, self.bernoulli_parameters.max() - self.bernoulli_parameters)


def train(environment: StatelessEnvironment, agent: Agent, n_steps: int) -> tuple[list[float], list[float]]:
    regret = 0.0
    regrets = []

    expected_regrets = []

    for _ in range(n_steps):
        action = agent.choose_action()
        reward = environment.get_reward(action)

        agent.update(action, reward)

        regret += environment.get_regret(action)
        regrets.append(regret)

        if isinstance(agent, PolicyAgent):
            expected_regret = environment.get_expected_regret(agent.get_policy())
            expected_regrets.append(expected_regret)

    return regrets, expected_regrets