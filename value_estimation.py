from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass
from typing import Optional, Callable
from contextlib import contextmanager, ExitStack
from math import pow
from collections import defaultdict


import numpy as np


# transition_probabilities.shape = (#states, #actions, #states), transition_probabilities[i, j, k] is the probability of transitioning from state j into state k after picking action i
# rewards.shape = (#states, #actions), rewards[i, j] is the reward for picking action j when in state i

@dataclass
class MarkovDecisionProcess:
    transition_probabilities: np.ndarray
    rewards: np.ndarray

    discount: float = 1.0

    def __post_init__(self):
        assert self.transition_probabilities.ndim == 3
        assert self.transition_probabilities.shape[0] == self.transition_probabilities.shape[2]

        assert self.rewards.ndim == 2
        assert self.rewards.shape[0] == self.transition_probabilities.shape[2]
        assert self.rewards.shape[1] == self.transition_probabilities.shape[1]

    @property
    def n_states(self):
        return self.transition_probabilities.shape[0]

    @property
    def n_actions(self):
        return self.transition_probabilities.shape[1]

    def __eq__(self, other) -> bool:
        if not isinstance(other, MarkovDecisionProcess):
            return NotImplemented

        return (
            np.array_equal(self.transition_probabilities, other.transition_probabilities) and
            np.array_equal(self.rewards, other.rewards) and
            self.discount == other.discount
        )

# action_probabilities.shape = (#states, #actions), action_probabilities[i, j] is the probability of picking action j when in state i

@dataclass
class Policy:
    action_probabilities: np.ndarray

    def __post_init__(self):
        assert self.action_probabilities.ndim == 2

    @property
    def n_action(self):
        return self.action_probabilities.shape[1]

    @property
    def n_states(self):
        return self.action_probabilities.shape[0]

    def __eq__(self, other) -> bool:
        if not isinstance(other, Policy):
            return NotImplemented
        return np.array_equal(self.action_probabilities, other.action_probabilities)


def make_uniform_policy(mdp: MarkovDecisionProcess) -> Policy:
    action_probabilities = np.full(shape=(mdp.n_states, mdp.n_actions), fill_value=1.0/mdp.n_actions)
    return Policy(action_probabilities)


def calculate_state_transition_matrix(mdp: MarkovDecisionProcess, policy: Policy) -> np.ndarray:
    return np.einsum("sa, saz->sz", policy.action_probabilities, mdp.transition_probabilities)


def calculate_value_function_iterative(mdp: MarkovDecisionProcess, policy: Policy, max_iterations: int=100) -> np.ndarray:
    assert mdp.n_states == policy.n_states
    assert mdp.n_actions == policy.n_action

    T = calculate_state_transition_matrix(mdp, policy)

    def value_operator(v):

        immediate_rewards = np.einsum("sa,sa->s", policy.action_probabilities, mdp.rewards)
        future_rewards = T @ v

        return immediate_rewards + mdp.discount * future_rewards

    v = np.zeros(shape=mdp.n_states)
    for _ in range(max_iterations):
        new_v = value_operator(v)

        if np.array_equal(v, new_v):
            break

        v = new_v

    return v


class Environment(ABC):
    def __init__(self) -> None:
        self.generator = np.random.default_rng()

    @abstractproperty
    def state(self) -> Optional[int]:
        raise NotImplementedError()

    @abstractmethod
    def reset(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def update(self, action: int) -> tuple[Optional[int], float]:
        raise NotImplementedError()


class MDPEnvironment(Environment):
    def __init__(self, mdp: MarkovDecisionProcess, initial_state: int = 0, terminal_states: Optional[set[int]] = None) -> None:
        super().__init__()

        assert 0 <= initial_state < mdp.n_states
        self.mdp = mdp

        self.initial_state = initial_state
        self.terminal_states = terminal_states if terminal_states is not None else {}

        self._state = initial_state

    @property
    def state(self) -> int:
        return self._state

    def reset(self) -> int:
        self._state = self.initial_state
        return self.state

    def update(self, action: int) -> tuple[Optional[int], float]:
        assert 0 <= action < self.mdp.n_actions

        if self._state is None:
            return None, 0.0

        reward = self.mdp.rewards[self.state, action]

        p = self.mdp.transition_probabilities[self.state, action]
        new_state = self.generator.choice(np.arange(self.mdp.n_states), p=p)

        self._state = new_state if new_state not in self.terminal_states else None

        return self.state, reward  


# class Episode(ABC):
#     @abstractmethod
#     def update(self, state, action, reward) -> None:
#         raise NotImplementedError()

#     @abstractmethod
#     def __enter__(self) -> 'Episode':
#         raise NotImplementedError()

#     @abstractmethod
#     def __exit__(self, *args) -> None:
#         raise NotImplementedError()


class ValueEstimator(ABC):
    def __init__(self, n_states: int, n_actions: int, discount: float) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.discount = discount

        self.n_episodes = 0

    @abstractproperty
    def value_function_estimate(self) -> np.ndarray:
        raise NotImplementedError()

    @abstractproperty
    def action_value_function_estimate(self) -> np.ndarray:    
        raise NotImplementedError()

    @abstractmethod
    def episode(self) -> Callable[[int, int, float], None]:
        raise NotImplementedError()


class OfflineMonteCarloEstimator(ValueEstimator):
    def __init__(self, n_states: int, n_actions: int, discount: float) -> None:
        super().__init__(n_states, n_actions, discount)

        self.total_action_values = np.zeros(shape=(n_states, n_actions))
        self.n_visits = np.zeros(shape=(n_states, n_actions), dtype=np.int32)

    @property
    def value_function_estimate(self) -> np.ndarray:
        return self.total_action_values.sum(axis=-1) / np.maximum(1, self.n_visits.sum(axis=-1))

    @property
    def action_value_function_estimate(self) -> np.ndarray:
        return self.total_action_values / np.maximum(1, self.n_visits)

    @contextmanager
    def episode(self) -> Callable[[int, int, float], None]:
        self.n_episodes += 1
        
        update_history = []
        def update(state, action, reward):
            update_history.append((state, action, reward))

        yield update

        G = 0
        for state, action, reward in reversed(update_history):
            G = reward + self.discount * G
            self.total_action_values[state, action] += G
            self.n_visits[state, action] += 1


class LearningRateSchedule(ABC):
    @abstractmethod
    def __call__(self, episode_number: int) -> float:
        raise NotImplementedError()


class ConstantLearningRate(LearningRateSchedule):
    def __init__(self, lr: float) -> None:
        self.lr = lr

    def __call__(self, episode_number: int) -> float:
        return self.lr


class LinearLearningRate(LearningRateSchedule):
    def __init__(self, start_lr: float, end_lr: float, n_episodes: int) -> None:
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.n_episodes = n_episodes

    def __call__(self, episode_number: int) -> float:
        alpha = min(1.0, episode_number / self.n_episodes)
        return self.start_lr * (1 - alpha) + self.end_lr * alpha


class OnlineMonteCarloEstimator(ValueEstimator):
    def __init__(self, n_states: int, n_actions: int, discount: float, lr: LearningRateSchedule) -> None:
        super().__init__(n_states, n_actions, discount)

        self.lr = lr
        self.v = np.zeros(shape=n_states)
        self.q = np.zeros(shape=(n_states, n_actions))

    @property
    def value_function_estimate(self) -> np.ndarray:
        return self.v

    @property
    def action_value_function_estimate(self) -> np.ndarray:
        return self.q

    @contextmanager
    def episode(self) -> Callable[[int, int, float], None]:
        self.n_episodes += 1
        lr = self.lr(self.n_episodes)
        
        update_history = []

        def update(state, action, reward):
            update_history.append((state, action, reward))

        yield update

        G = 0
        for state, action, reward in reversed(update_history):
            G = reward + self.discount * G
            self.v[state] += lr * (G - self.v[state])
            self.q[state, action] += lr * (G - self.q[state, action])


class TemporalDifferenceEstimator(ValueEstimator):
    def __init__(self, n_states: int, n_actions: int, discount: float, initial_g_estimate: float, lr: LearningRateSchedule) -> None:
        super().__init__(n_states, n_actions, discount)

        self.lr = lr
        self.v = np.full(shape=n_states, fill_value=initial_g_estimate)
        self.q = np.full(shape=(n_states, n_actions), fill_value=initial_g_estimate)

    @property
    def value_function_estimate(self) -> np.ndarray:
        return self.v

    @property
    def action_value_function_estimate(self) -> np.ndarray:
        return self.q

    @contextmanager
    def episode(self) -> Callable[[int, int, float], None]:
        self.n_episodes += 1
        lr = self.lr(self.n_episodes)
        
        previous_state = None
        previous_action = None
        previous_reward = None

        def update(state, action, reward):
            nonlocal previous_state, previous_action, previous_reward

            if previous_state is not None:
                self.v[previous_state] += lr * (previous_reward + self.discount * self.v[state] - self.v[previous_state])
                self.q[previous_state, previous_action] += lr * (previous_reward + self.discount * self.q[state, action] - self.q[previous_state, previous_action])

            previous_state = state
            previous_action = action
            previous_reward = reward

        yield update

        if previous_state is not None:
            self.v[previous_state] += lr * (previous_reward - self.v[previous_state])
            self.q[previous_state, previous_action] += lr * (previous_reward - self.q[previous_state, previous_action])       


class MixingMultiStepEstimator(ValueEstimator):
    def __init__(self, n_states: int, n_actions: int, discount: float, n_steps: int, lambda_factor: float, initial_g_estimate: float, lr: LearningRateSchedule) -> None:
        super().__init__(n_states, n_actions, discount)

        self.n_steps = n_steps
        self.lambda_factor = lambda_factor

        self.lr = lr
        self.v = np.full(shape=n_states, fill_value=initial_g_estimate)
        # self.q = np.full(shape=(n_states, n_actions), fill_value=initial_g_estimate)

    @property
    def value_function_estimate(self) -> np.ndarray:
        return self.v

    @property
    def action_value_function_estimate(self) -> np.ndarray:
        raise NotImplementedError()
        # return self.q

    @contextmanager
    def episode(self) -> Callable[[int, int, float], None]:
        self.n_episodes += 1
        lr = self.lr(self.n_episodes)

        trajectory: list[tuple[int, int, float]] = []

        def calculate_G(bootstrap_last: bool):
            assert len(trajectory) > 0 

            s_last, _, r_last = trajectory[-1]
            if bootstrap_last:
                assert len(trajectory) == self.n_steps + 1
                G = self.v[s_last]
            else:
                G = self.lambda_factor * r_last + (1 - self.lambda_factor) * self.v[s_last]

            next_s = s_last
            for s, a, r in reversed(trajectory[:-1]):
                G = r + self.discount * (self.lambda_factor * G + (1 - self.lambda_factor) * self.v[next_s])
                next_s = s

            return G


        def update(state, action, reward):
            nonlocal trajectory
            trajectory.append((state, action, reward))

            if len(trajectory) >= self.n_steps + 1:
                G = calculate_G(True)
                s, a, r = trajectory.pop(0)

                self.v[s] += lr * (G - self.v[s])

        yield update

        while len(trajectory) > 0:
            G = calculate_G(False)
            s, a, r = trajectory.pop(0)

            self.v[s] += lr * (G - self.v[s])


def run(environment: Environment, policy: Policy, value_estimators: dict[str, ValueEstimator], n_episodes: int, real_value_function: Optional[np.ndarray] = None):

    loss_histories = defaultdict(list)

    for _ in range(n_episodes):
        environment.reset()

        with ExitStack() as stack:

            ve_update_functions = []
            for value_estimator in value_estimators.values():
                ve_update_functions.append(stack.enter_context(value_estimator.episode()))

            while environment.state is not None:
                old_state = environment.state

                action = environment.generator.choice(np.arange(policy.n_action), p=policy.action_probabilities[old_state])
                _, reward = environment.update(action)

                for ve_update in ve_update_functions:
                    ve_update(old_state, action, reward)

        if real_value_function is not None:
            for name, value_estimator in value_estimators.items():
                loss = np.linalg.norm(value_estimator.value_function_estimate - real_value_function)**2
                loss_histories[name].append(loss)

    return loss_histories