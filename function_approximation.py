from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Callable
import multiprocessing as mp

import numpy as np

import gymnasium as gym
from gymnasium.spaces import Discrete, Box
from tqdm import tqdm

from model_free_control import Schedule, LinearSchedule, ConstantSchedule, ReciprocalSchedule, ExponentialSchedule, StepInfo


class InputOutputAdapter(ABC):
    @abstractmethod
    def convert_to(self, x):
        raise NotImplementedError
    
    @abstractmethod
    def convert_from(self, x):
        raise NotImplementedError
    

class IdentityAdapter(ABC):
    def convert_to(self, x):
        return x
    
    def convert_from(self, x):
        return x


class ActionValueEstimator(ABC):
    @abstractmethod
    def action_values(self, observation) -> np.array:
        raise NotImplementedError

    @abstractmethod
    def update(self, observation, action, target: float, alpha: float):
        raise NotImplementedError
    
    def value(self, observation, action: int) -> float:
        return self.action_values(observation)[action]
    
    def argmax_action(self, observation):
        return self.action_values(observation).argmax()
    
    def max_action_value(self, observation) -> float:
        return self.action_values(observation).max()


class ActionValueEstimatorAdapterWrapper(ActionValueEstimator):
    def __init__(self, av_estimator: ActionValueEstimator, observation_adapter: InputOutputAdapter | None, action_adapter: InputOutputAdapter | None) -> None:
        super().__init__()

        self.observation_adapter = observation_adapter if observation_adapter is not None else IdentityAdapter()
        self.action_adapter = action_adapter if action_adapter is not None else IdentityAdapter()
        self.av_estimator = av_estimator

    def action_values(self, observation) -> np.array:
        observation = self.observation_adapter.convert_to(observation)
        return self.av_estimator.action_values(observation)

    def update(self, observation, action, target: float, alpha: float):
        observation = self.observation_adapter.convert_to(observation)
        action = self.action_adapter.convert_to(action)
        self.av_estimator.update(observation, action, target, alpha)

    def value(self, observation, action) -> float:
        observation = self.observation_adapter.convert_to(observation)
        action = self.action_adapter.convert_to(action)
        return self.av_estimator.value(observation, action)
    
    def argmax_action(self, observation):
        observation = self.observation_adapter.convert_to(observation)
        action = self.av_estimator.argmax_action(observation)
        return self.action_adapter.convert_from(action)  

    def max_action_value(self, observation) -> float:
        observation = self.observation_adapter.convert_to(observation)
        return self.av_estimator.max_action_value(observation)


class TabularEstimator(ActionValueEstimator):
    def __init__(self, n_observations: int, n_actions: int, initial_value: float = 0.0) -> None:
        super().__init__()

        self.n_observations = n_observations
        self.n_actions = n_actions

        self.q = np.full(shape=(n_observations, n_actions), fill_value=initial_value)

    def action_values(self, observation) -> np.array:
        return self.q[observation]

    def update(self, observation: int, action: int, target: float, alpha: float):
        self.q[observation, action] += alpha * (target - self.q[observation, action])


class LinearActionOutEstimator(ActionValueEstimator):
    def __init__(self, n_features: int, n_actions: int) -> None:
        super().__init__()

        self.n_features = n_features
        self.n_actions = n_actions

        self.w = np.zeros(shape=(n_actions, n_features), dtype=np.float32)

    def action_values(self, observation_features: np.array) -> np.array:
        return self.w @ observation_features
    
    def update(self, observation_features: np.array, action: int, target: float, alpha: float) -> float:
        self.w[action] += alpha * (target - self.value(observation_features, action)) * observation_features

    def value(self, observation_features: np.array, action: int) -> float:
        return self.w[action] @ observation_features


class OneHotAdapter(InputOutputAdapter):
    def __init__(self, size: int, dtype: np.dtype = np.float32) -> None:
        self.size = size
        self.dtype = dtype

    def convert_to(self, x: int) -> np.array:
        v = np.zeros(self.size, dtype=self.dtype)
        v[x] = 1.0
        return v
    
    def convert_from(self, x: np.array) -> int:
        return x.argmax()
    

class FlattenDiscreteAdapter(InputOutputAdapter):
    def __init__(self, space: gym.spaces.Tuple | Discrete) -> None:
        self.spaces = (space,) if not isinstance(space, (tuple, gym.spaces.Tuple)) else space
        assert len(self.spaces) > 0

    def convert_to(self, x):
        if len(self.spaces) == 1:
            return x

        assert len(self.spaces) == len(x) > 0

        index = 0
        for space, y in zip(self.spaces, x):
            start = 0 if space.start is None else space.start
            assert start <= y < start + space.n
            
            index *= space.n
            index += y - start

        return index
    
    def convert_from(self, x):
        if len(self.spaces) == 1:
            return x

        indices = []
        for space in reversed(self.spaces):
            start = 0 if space.start is None else space.start
            index = start + (x % space.n)
            indices.append(index)

            x //= space.n

        return tuple(reversed(indices))

    @property
    def dim(self):
        product = 1
        for space in self.spaces:
            product *= space.n
        return product


class TileCodingAdapter(InputOutputAdapter):
    def __init__(self, space: Box, n_divisions: int, n_shifts: int = 1, dtype: np.dtype = np.float32) -> None:
        assert isinstance(space, Box)
        self.space = space
        self.n_divisions = n_divisions
        self.n_shifts = n_shifts
        self.dtype = dtype

        self.lengths = self.space.high - self.space.low
        self.shift = self.lengths / (n_divisions * n_shifts)

        self.flatten_adapter = FlattenDiscreteAdapter((Discrete(n_divisions),) * space.low.shape[0])

    def convert_to(self, x: np.array) -> np.array:
        v = np.zeros(self.size, dtype=self.dtype)

        for i in range(self.n_shifts):
            index = self.convert_to_index(x + i * self.shift)
            v[i * self.n_divisions + index] = 1.0
        
        return v

    def convert_to_index(self, x: np.array) -> int:
        x = np.clip(x, self.space.low, self.space.high)

        indices = (x - self.space.low) / self.lengths * self.n_divisions
        indices = np.minimum(np.floor(indices).astype(np.int64), self.n_divisions-1)

        index = self.flatten_adapter.convert_to(indices)
        return index
    
    def convert_from(self, x):
        raise NotImplementedError
    
    @property
    def size(self):
        return self.n_shifts * self.flatten_adapter.dim
    

@dataclass
class StepInfo:
    observation: int
    action: int
    reward: int


class MFCAgent(ABC):
    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self.n_episodes = 0

        self.rng = rng if rng is not None else np.random.default_rng()

    @abstractmethod
    def start_episode(self):
        self.n_episodes += 1

    @abstractmethod
    def end_episode(self, terminated: bool):
        raise NotImplementedError()

    @abstractmethod
    def get_action(self, observation, eval: bool = False):
        raise NotImplementedError()

    @abstractmethod
    def update(self, step: StepInfo):
        raise NotImplementedError()
    

class ActionValueEstimateAgent(MFCAgent):
    def __init__(self, av_estimator: ActionValueEstimator, n_actions: int, gamma: float, alpha_schedule: Schedule, epsilon_schedule: Schedule, rng: np.random.Generator | None = None) -> None:
        super().__init__()

        self.n_actions = n_actions
        self.gamma = gamma

        self.alpha_schedule = alpha_schedule
        self.epsilon_schedule = epsilon_schedule

        self.av_estimator = av_estimator

    def start_episode(self):
        super().start_episode()

    def end_episode(self, terminated: bool):
        pass

    def get_action(self, observation, eval: bool = False):
        epsilon = self.epsilon_schedule(self.n_episodes)

        if not eval and self.rng.random() <= epsilon:
            return self.rng.integers(0, self.n_actions)
        else:
            return self.av_estimator.argmax_action(observation)


class MonteCarloAgent(ActionValueEstimateAgent):
    def __init__(self, av_estimator: ActionValueEstimator, n_actions: int, gamma: float, alpha_schedule: Schedule, epsilon_schedule: Schedule, rng: np.random.Generator | None = None) -> None:
        super().__init__(av_estimator, n_actions, gamma, alpha_schedule, epsilon_schedule, rng)

        self.episode_history: list[StepInfo] = []

    def start_episode(self):
        super().start_episode()
        self.episode_history.clear()

    def end_episode(self, terminated: bool):
        alpha = self.alpha_schedule(self.n_episodes)

        G = 0
        for step in reversed(self.episode_history):
            G = step.reward + self.gamma * G
            self.av_estimator.update(step.observation, step.action, G, alpha)

        self.episode_history.clear()
        
    def update(self, step: StepInfo):
        self.episode_history.append(step)


class OnlineActionValueEstimateAgent(ActionValueEstimateAgent):
    def __init__(self, av_estimator: ActionValueEstimator, n_actions: int, gamma: float, alpha_schedule: Schedule, epsilon_schedule: Schedule, bootstrap_value_fn: Callable, rng: np.random.Generator | None = None) -> None:
        super().__init__(av_estimator, n_actions, gamma, alpha_schedule, epsilon_schedule, rng)

        self.bootstrap_value_fn = bootstrap_value_fn
        self.last_step: StepInfo = None

    def start_episode(self):
        super().start_episode()
        self.last_step = None

    def end_episode(self, terminated: bool):
        if self.last_step is not None and terminated:
            self._update_av_estimator(self.last_step, self.last_step.reward)

        self.last_step = None

    def update(self, step: StepInfo):
        if self.last_step is not None:
            bootstrap_value = self.bootstrap_value_fn(self.av_estimator, step, self.epsilon_schedule(self.n_episodes))
            self._update_av_estimator(self.last_step, self.last_step.reward + self.gamma * bootstrap_value)

        self.last_step = step
    
    def _update_av_estimator(self, step: StepInfo, target: float):
        alpha = self.alpha_schedule(self.n_episodes)
        self.av_estimator.update(step.observation, step.action, target, alpha)


class MixingMultiStepAgent(ActionValueEstimateAgent):
    def __init__(self, av_estimator: ActionValueEstimator, n_actions: int, gamma: float, alpha_schedule: Schedule, epsilon_schedule: Schedule, lambda_: float, n: int, bootstrap_value_fn: Callable, rng: np.random.Generator | None = None) -> None:
        super().__init__(av_estimator, n_actions, gamma, alpha_schedule, epsilon_schedule, rng)

        self.bootstrap_value_fn = lambda func_est, step, *args: bootstrap_value_fn(func_est, step, *args) if step is not None else 0.0

        self.lambda_ = lambda_
        self.n = n
        self.trajectory = []

    def start_episode(self):
        super().start_episode()
        self.trajectory.clear()

    def end_episode(self, terminated: bool):
        if terminated:
            self.trajectory.append(None)

        while len(self.trajectory) > 1:
            self._update_step()

        self.trajectory.clear()

    def update(self, step: StepInfo):
        self.trajectory.append(step)

        if len(self.trajectory) >= self.n + 1:
            self._update_step()

    def _update_step(self):
        assert len(self.trajectory) >= 2
        alpha = self.alpha_schedule(self.n_episodes)
        epsilon = self.epsilon_schedule(self.n_episodes)

        last_step = self.trajectory[-1]
        G = self.bootstrap_value_fn(self.av_estimator, last_step, epsilon)

        for step in reversed(self.trajectory[:-1]):
            G = step.reward + self.gamma * ( (1 - self.lambda_) * self.bootstrap_value_fn(self.av_estimator, last_step, epsilon) + self.lambda_ * G)
            last_step = step

        self.av_estimator.update(last_step.observation, last_step.action, G, alpha)
        self.trajectory.pop(0)


def sarsa_bootstrap_value(av_estimator: ActionValueEstimator, future_step: StepInfo, epsilon: float=0):
    return av_estimator.value(future_step.observation, future_step.action)


def q_learning_bootstrap_value(av_estimator: ActionValueEstimator, future_step: StepInfo, epsilon: float=0):
    return av_estimator.max_action_value(future_step.observation)
    

def expected_sarsa_bootstrap_value(av_estimator: ActionValueEstimator, future_step: StepInfo, epsilon: float=0):
    action_values = av_estimator.action_values(future_step.observation)
    assert isinstance(action_values, np.ndarray)

    argmax_action = action_values.argmax()

    policy = np.full_like(action_values, fill_value=epsilon / action_values.shape[0])
    policy[argmax_action] += 1 - epsilon

    return action_values @ policy


class EligibilityTracesAgent(MFCAgent):
    def __init__(self, n_features: int, n_actions: int, gamma: float, lambda_: float, alpha_schedule: Schedule, epsilon_schedule: Schedule, feature_adapter: InputOutputAdapter, rng: np.random.Generator | None = None) -> None:
        super().__init__(rng)

        self.n_features = n_features
        self.n_actions = n_actions

        self.e = np.zeros(shape=(n_actions, n_features), dtype=np.float32)
        self.w = np.zeros(shape=(n_actions, n_features), dtype=np.float32)

        self.gamma = gamma
        self.lambda_ = lambda_

        self.alpha_schedule = alpha_schedule
        self.epsilon_schedule = epsilon_schedule

        self.feature_adapter = feature_adapter

        self.last_step = None
        self.last_features = None

    def start_episode(self):
        super().start_episode()
        self.e.fill(0.0)
        self.last_step = None
        self.last_features = None

    def end_episode(self, terminated: bool):
        if self.last_features is not None and terminated:
            alpha = self.alpha_schedule(self.n_episodes)

            self.e *= self.gamma * self.lambda_
            self.e[self.last_step.action] += self.last_features

            td_error = self.last_step.reward - (self.w[self.last_step.action] @ self.last_features)
            self.w += alpha * td_error * self.e

        self.last_step = None
        self.last_features = None

    def update(self, step: StepInfo):
        features = self.feature_adapter.convert_to(step.observation)

        if self.last_step is not None:
            alpha = self.alpha_schedule(self.n_episodes)

            self.e *= self.gamma * self.lambda_
            self.e[self.last_step.action] += self.last_features

            td_error = self.last_step.reward + self.gamma * (self.w[step.action] @ features) - (self.w[self.last_step.action] @ self.last_features)
            self.w += alpha * td_error * self.e

        self.last_step = step
        self.last_features = features
        
    
    def get_action(self, observation, eval: bool = False):
        epsilon = self.epsilon_schedule(self.n_episodes)

        if not eval and self.rng.random() <= epsilon:
            return self.rng.integers(0, self.n_actions)
        else:
            return (self.w @ self.feature_adapter.convert_to(observation)).argmax()

    
def train_episode(environment: gym.Env, agent: MFCAgent):
    observation, info = environment.reset()
    agent.start_episode()

    total_reward = 0.0
    episode_length = 0
    while True:
        action = agent.get_action(observation)

        next_observation, reward, terminated, truncated, info = environment.step(action)
        total_reward += reward

        step = StepInfo(observation, action, reward)
        agent.update(step)

        observation = next_observation
        episode_length += 1

        if terminated or truncated:
            break

    agent.end_episode(terminated)
    return total_reward, episode_length, truncated


def train(environment: gym.Env, agent: MFCAgent, n_episodes: int):
    rewards = np.zeros(n_episodes, dtype=np.float32)
    episode_lengths = np.zeros(n_episodes, dtype=np.int64)
    n_truncated = 0
    n_terminated = 0

    for i in tqdm(range(n_episodes)):
        reward, episode_length, truncated = train_episode(environment, agent)

        rewards[i] = reward
        episode_lengths[i] = episode_length

        if truncated:
            n_truncated += 1
        else:
            n_terminated += 1

    return rewards, episode_lengths, n_terminated, n_truncated
