from dataclasses import dataclass, field

import numpy as np


# transition_probabilities.shape = (#actions, #states, #states), transition_probabilities[i, j, k] is the probability of transitioning from state j into state k after picking action i
# the rewards probability function is simplified a bit. We simply get a reward rewards[i] for being in state i

@dataclass
class MarkovDecisionProcess:
    transition_probabilities: np.ndarray
    rewards: np.ndarray

    discount: float = 1.0

    def __post_init__(self):
        assert self.transition_probabilities.ndim == 3
        assert self.transition_probabilities.shape[1] == self.transition_probabilities.shape[2]

        assert self.rewards.ndim == 1
        assert self.rewards.shape[0] == self.transition_probabilities.shape[2]

    @property
    def n_states(self):
        return self.transition_probabilities.shape[2]

    @property
    def n_actions(self):
        return self.transition_probabilities.shape[0]

    def __eq__(self, other: 'MarkovDecisionProcess') -> bool:
        return (
            np.array_equal(self.transition_probabilities, other.transition_probabilities) and
            np.array_equal(self.rewards, other.rewards) and
            self.discount == other.discount and
            self.initial_state == other.initial_state
        )


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

    def __eq__(self, other: 'Policy') -> bool:
        return np.array_equal(self.action_probabilities, other.action_probabilities)


def calculate_state_transition_matrix(mdp: MarkovDecisionProcess, policy: Policy) -> np.ndarray:
    return np.einsum("ij,jik->ik", policy.action_probabilities, mdp.transition_probabilities)


# value_function.shape = (#states), value_function[i] is the value function associated to state i
def calculate_value_function(mdp: MarkovDecisionProcess, policy: Policy) -> np.ndarray:
    assert mdp.n_states == policy.n_states
    assert mdp.n_actions == policy.n_action

    T = calculate_state_transition_matrix(mdp, policy)
    return np.linalg.pinv(np.identity(mdp.n_states) - mdp.discount * T) @ T @ mdp.rewards


def calculate_value_function_iterative(mdp: MarkovDecisionProcess, policy: Policy, max_iterations: int=100) -> np.ndarray:
    assert mdp.n_states == policy.n_states
    assert mdp.n_actions == policy.n_action

    T = calculate_state_transition_matrix(mdp, policy)
    v = np.zeros(shape=mdp.n_states)

    for _ in range(max_iterations):
        v_new = T @ (mdp.rewards + mdp.discount * v)

        if np.array_equal(v, v_new):
            break

        v = v_new

    return v


# action_value_function.shape = (#states, #action) with action_value_function[i, j] is the action value (q) for taking action j when in state i
def calculate_action_value_function(mdp: MarkovDecisionProcess, value_function: np.ndarray) -> np.ndarray:
    assert value_function.ndim == 1 and value_function.shape[0] == mdp.n_states

    action_value_function_transposed = mdp.transition_probabilities @ (mdp.rewards + mdp.discount * value_function)
    return action_value_function_transposed.T


def calculate_action_value_function_from_policy(mdp: MarkovDecisionProcess, policy: Policy) -> np.ndarray:
    value_function = calculate_value_function(mdp, policy)
    return calculate_action_value_function(mdp, value_function)


def calculate_greedy_policy(action_value_function) -> np.ndarray:
    is_max_action = action_value_function == action_value_function.max(axis=-1, keepdims=True)
    return Policy(is_max_action / is_max_action.sum(axis=-1, keepdims=True))


def policy_iteration(mdp: MarkovDecisionProcess, initial_policy: Policy, max_iterations: int) -> Policy:
    policy = initial_policy

    for _ in range(max_iterations):
        action_value_function = calculate_action_value_function_from_policy(mdp, policy)
        new_policy = calculate_greedy_policy(action_value_function)

        if policy == new_policy:
            break
        policy = new_policy
    
    return policy


def calculate_optimal_value_function(mdp: MarkovDecisionProcess, max_iterations: int) -> np.ndarray:

    v = np.zeros(shape=mdp.n_states)

    for _ in range(max_iterations):
        new_v = (mdp.transition_probabilities @ (mdp.rewards + mdp.discount * v)).max(axis=0)

        if np.array_equal(v, new_v):
            break

        v = new_v

    return v