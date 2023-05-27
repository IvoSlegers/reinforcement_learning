from typing import Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
from math import exp, pow

import numpy as np

import gymnasium as gym
from gymnasium.spaces import Discrete

from tqdm import tqdm


def check_tuple_type(spaces, type):
    if isinstance(spaces, type):
        spaces = (spaces, )
    assert isinstance(spaces, (tuple, gym.spaces.Tuple))
    for space in spaces:
        isinstance(space, type)

    return spaces


def discrete_flatten_dim(spaces: tuple[Discrete,...] | Discrete) -> int:
    spaces = check_tuple_type(spaces, Discrete)
    assert len(spaces) > 0
    product = 1
    for space in spaces:
        product *= space.n
    return product

def discrete_flatten(spaces: tuple[Discrete,...] | Discrete, x: tuple[int,...]) -> int:
    spaces = check_tuple_type(spaces, Discrete)
    x = check_tuple_type(x, int)
    assert len(spaces) == len(x) > 0

    index = 0
    for space, y in zip(spaces, x):
        start = 0 if space.start is None else space.start
        assert start <= y < start + space.n
        
        index *= space.n
        index += y - start

    return index

def discrete_unflatten(spaces: tuple[Discrete,...] | Discrete, x: int):
    if isinstance(spaces, Discrete):
        return x

    spaces = check_tuple_type(spaces, Discrete)
    assert len(spaces) > 0

    indices = []
    for space in reversed(spaces):
        start = 0 if space.start is None else space.start
        index = start + (x % space.n)
        indices.append(index)

        x //= space.n

    return tuple(reversed(indices))


class Schedule(ABC):
    @abstractmethod
    def __call__(self, episode: int) -> float:
        raise NotImplementedError()


class ConstantSchedule(Schedule):
    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self, episode: int) -> float:
        return self.value


class ReciprocalSchedule(Schedule):
    def __init__(self, start_value: float, exponential: float = 1.0) -> None:
        self.start_value = start_value
        self.exponential = exponential

    def __call__(self, episode: int) -> float:
        return self.start_value / max(1, pow(episode, self.exponential))


class ExponentialSchedule(Schedule):
    def __init__(self, start_value: float, exponent: float) -> None:
        self.start_value = start_value
        self.exponent = exponent

    def __call__(self, episode: int) -> float:
        return self.start_value * exp(-self.exponent * episode)


class OffsetWrapper(Schedule):
    def __init__(self, schedule: Schedule, offset: float) -> None:
        self.schedule = schedule
        self.offset = offset

    def __call__(self, episode: int) -> float:
        return self.offset + self.schedule(episode)


class LinearSchedule(Schedule):
    def __init__(self, start: float, end: float, n_episodes: int) -> None:
        self.start = start
        self.end = end
        self.n_episodes = n_episodes

    def __call__(self, episode: int) -> float:
        alpha = min(self.n_episodes, episode) / self.n_episodes
        return self.start * (1 - alpha) + self.end * alpha


@dataclass
class StepInfo:
    observation: int
    action: int
    reward: int


class MFCAgent(ABC):
    def __init__(self, n_observations: int, n_actions: int, rng: Optional[np.random.Generator] = None) -> None:
        self.n_observations = n_observations
        self.n_actions = n_actions

        self.n_episodes = 0

        self.rng = rng if rng is not None else np.random.default_rng()

    @abstractmethod
    def start_episode(self):
        self.n_episodes += 1

    @abstractmethod
    def end_episode(self):
        raise NotImplementedError()

    @abstractmethod
    def get_action(self, observation: int, eval: bool = False):
        raise NotImplementedError()

    @abstractmethod
    def update(self, last_step: StepInfo, step: StepInfo | None):
        raise NotImplementedError()


class ActionValueEstimateAgent(MFCAgent):
    def __init__(self, n_observations: int, n_actions: int, gamma: float, alpha_schedule: Schedule, epsilon_schedule: Schedule) -> None:
        super().__init__(n_observations, n_actions)

        self.gamma = gamma
        self.alpha_schedule = alpha_schedule
        self.epsilon_schedule = epsilon_schedule

        self.q = np.zeros(shape=(n_observations, n_actions))

    def start_episode(self):
        super().start_episode()

    def end_episode(self):
        pass

    def get_action(self, observation: int, eval: bool = False):
        epsilon = self.epsilon_schedule(self.n_episodes)

        if not eval and self.rng.random() <= epsilon:
            return self.rng.integers(0, self.n_actions)
        else:
            return self.q[observation].argmax()

    def _update_q(self, step: StepInfo, future_q: float):
        alpha = self.alpha_schedule(self.n_episodes)

        target = step.reward + self.gamma * future_q - self.q[step.observation, step.action]
        self.q[step.observation, step.action] += alpha * target


class SarsaAgent(ActionValueEstimateAgent):
    def update(self, last_step: StepInfo, step: StepInfo | None):
        future_q = self.q[step.observation, step.action] if step is not None else 0.0
        self._update_q(last_step, future_q)


class QLearningAgent(ActionValueEstimateAgent):
    def update(self, last_step: StepInfo, step: StepInfo | None):
        future_q = self.q[step.observation].max() if step is not None else 0.0
        self._update_q(last_step, future_q)        


class ExpectedSarsa(ActionValueEstimateAgent):
    def policy(self, observation: int):
        epsillon = self.epsilon_schedule(self.n_episodes)

        action_values = self.q[observation]
        argmax_action = action_values.argmax()

        policy = np.full_like(action_values, fill_value=epsillon / action_values.size)
        policy[argmax_action] += (1-epsillon)

        return policy        

    def update(self, last_step: StepInfo, step: StepInfo | None):
        if step is not None:
            policy = self.policy(step.observation)
            future_q = np.dot(policy, self.q[step.observation])
        else:
            future_q = 0.0
        self._update_q(last_step, future_q)

class DoubleQLearningAgent(MFCAgent):
    def __init__(self, n_observations: int, n_actions: int, gamma: float, alpha_schedule: Schedule, epsilon_schedule: Schedule) -> None:
        super().__init__(n_observations, n_actions)

        self.gamma = gamma
        self.alpha_schedule = alpha_schedule
        self.epsilon_schedule = epsilon_schedule

        self.q1 = np.zeros(shape=(n_observations, n_actions))
        self.q2 = np.zeros(shape=(n_observations, n_actions))
        self.last_sar = None

    def start_episode(self):
        super().start_episode()
        self.last_sar = None

    def end_episode(self):
        pass

    def get_action(self, observation: int, eval: bool = False):
        epsillon = self.epsilon_schedule(self.n_episodes)

        if not eval and self.rng.random() <= epsillon:
            return self.rng.integers(0, self.n_actions)
        else:
            return (self.q1[observation] + self.q2[observation]).argmax()

    def update(self, last_step: StepInfo, step: StepInfo | None):
        alpha = self.alpha_schedule(self.n_episodes)

        use_first_q = self.rng.random() <= 0.5
        (q1_, q2_) = (self.q1, self.q2) if use_first_q else (self.q2, self.q1)

        if step is None:
            future_q = 0.0
        else:
            argmax_action = q1_[step.observation].argmax()
            future_q = q2_[step.observation, argmax_action]

        target = last_step.reward + self.gamma * future_q - q1_[last_step.observation, last_step.action]
        q1_[last_step.observation, last_step.action] += alpha * target


class OffPolicyQLearningExploration(QLearningAgent):
    def __init__(self, n_observations: int, n_actions: int, gamma: float, alpha_schedule: Schedule, epsilon_schedule: Schedule) -> None:
        super().__init__(n_observations, n_actions, gamma, alpha_schedule, epsilon_schedule)

        self.actions_taken = np.zeros(shape=(n_observations, n_actions), dtype=np.int64)

    def get_action(self, observation: int, eval: bool = False):
        epsillon = self.epsilon_schedule(self.n_episodes)

        if not eval and self.rng.random() <= epsillon:
            return self.actions_taken[observation].argmin()
        else:
            return self.q[observation].argmax()

    def update(self, last_step: StepInfo, step: StepInfo | None):
        super().update(last_step, step)

        self.actions_taken[last_step.observation, last_step.action] += 1


def train_episode(environment: gym.Env, agent: MFCAgent):
    observation, info = environment.reset()
    observation = discrete_flatten(environment.observation_space, observation)
    
    agent.start_episode()

    previous_step = None
    total_reward = 0.0
    while True:
        action = agent.get_action(observation)
        action_unflattend = discrete_unflatten(environment.action_space, action)

        next_observation, reward, terminated, truncated, info = environment.step(action_unflattend)
        next_observation = discrete_flatten(environment.observation_space, next_observation)
        total_reward += reward

        step = StepInfo(observation, action, reward)
        if previous_step is not None:
            agent.update(previous_step, step)

        previous_step = step
        observation = next_observation

        if terminated or truncated:
            if terminated:
                agent.update(step, None)
            break

    agent.end_episode()
    return total_reward, truncated


def train(environment: gym.Env, agent: MFCAgent, n_episodes: int):
    rewards = np.zeros(n_episodes, dtype=np.float32)
    n_truncated = 0
    n_terminated = 0

    for i in tqdm(range(n_episodes)):
        reward, truncated = train_episode(environment, agent)
        rewards[i] = reward

        if truncated:
            n_truncated += 1
        else:
            n_terminated += 1

    return rewards, n_terminated, n_truncated


def evaluate_episode(environment: gym.Env, agent: MFCAgent):
    observation, info = environment.reset()
    observation = discrete_flatten(environment.observation_space, observation)
    
    total_reward = 0.0
    while True:
        action = agent.get_action(observation, eval=True)
        action_unflattend = discrete_unflatten(environment.action_space, action)

        next_observation, reward, terminated, truncated, info = environment.step(action_unflattend)
        next_observation = discrete_flatten(environment.observation_space, next_observation)
        total_reward += reward

        observation = next_observation

        if terminated or truncated:
            break

    agent.end_episode()
    return total_reward, truncated


def evaluate(environment: gym.Env, agent: MFCAgent, n_episodes: int, progress_bar: bool=True):
    rewards = np.zeros(n_episodes, dtype=np.float32)
    n_truncated = 0
    n_terminated = 0

    progress_bar_fn = tqdm if progress_bar else lambda x: x

    for i in progress_bar_fn(range(n_episodes)):
        reward, truncated = train_episode(environment, agent)
        rewards[i] = reward

        if truncated:
            n_truncated += 1
        else:
            n_terminated += 1

    return rewards.mean(), rewards.std(), n_terminated, n_truncated