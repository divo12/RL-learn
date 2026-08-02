"""Lecture 1 mastery: MDPs, trajectories, values, and occupancy.

Conventions:
    S = number of states
    A = number of actions
    transitions[s, a, next_s] = p(next_s | s, a)
    rewards[s, a, next_s] = reward on that transition
    policy[s, a] = pi(a | s)

All TODOs are original practice exercises, not course-homework code.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def discounted_returns(
    rewards: FloatArray,
    dones: BoolArray,
    gamma: float,
) -> FloatArray:
    """Compute a return for every transition in a concatenated rollout.

    A true dones[t] means rewards[t] is terminal and rewards[t + 1], if any,
    belongs to a new episode. Inputs have shape (T,); output has shape (T,).
    Reject invalid shapes and gamma outside [0, 1].
    """
    # TODO(L1-returns): implement one backwards pass with terminal resets.
    raise NotImplementedError


def trajectory_log_probability(
    initial_probability: float,
    selected_action_probabilities: FloatArray,
    selected_transition_probabilities: FloatArray,
) -> float:
    """Return log p_theta(tau) for one sampled trajectory.

    selected_action_probabilities[t] is pi(a_t | s_t).
    selected_transition_probabilities[t] is p(s_{t+1} | s_t, a_t).
    Use log-space, validate equal one-dimensional shapes, and reject
    non-positive probabilities.
    """
    # TODO(L1-trajectory): implement the MDP trajectory factorization.
    raise NotImplementedError


def induced_policy_mdp(
    transitions: FloatArray,
    rewards: FloatArray,
    policy: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Return (P_pi, r_pi) for a fixed policy.

    Args:
        transitions: shape (S, A, S).
        rewards: shape (S, A, S).
        policy: shape (S, A), rows sum to one.

    Returns:
        P_pi: shape (S, S).
        r_pi: shape (S,), expected immediate reward from each state.
    """
    # TODO(L1-induced-mdp): marginalize actions and next states correctly.
    raise NotImplementedError


def evaluate_policy(
    transitions: FloatArray,
    rewards: FloatArray,
    policy: FloatArray,
    gamma: float,
) -> FloatArray:
    """Solve the Bellman expectation equations exactly.

    Return V satisfying V = r_pi + gamma * P_pi @ V.
    This exercise expects a linear solve, not iterative approximation.
    """
    # TODO(L1-policy-evaluation): form and solve the linear system.
    raise NotImplementedError


def action_values(
    transitions: FloatArray,
    rewards: FloatArray,
    values: FloatArray,
    gamma: float,
) -> FloatArray:
    """Compute Q(s,a) from a supplied V; return shape (S, A)."""
    # TODO(L1-q): take the expectation over next states.
    raise NotImplementedError


def advantages(policy: FloatArray, q_values: FloatArray) -> FloatArray:
    """Compute A(s,a) = Q(s,a) - V(s), with V induced by policy and Q."""
    # TODO(L1-advantage): preserve shape (S, A).
    raise NotImplementedError


def discounted_state_occupancy(
    transitions: FloatArray,
    policy: FloatArray,
    initial_distribution: FloatArray,
    gamma: float,
) -> FloatArray:
    """Compute normalized discounted state occupancy.

    d_pi(s) = (1-gamma) * sum_{t>=0} gamma^t P(s_t=s | pi).
    Use an exact linear solve and a consistent row-vector convention.
    Require gamma < 1 and return a vector summing to one.
    """
    # TODO(L1-occupancy): derive the transpose/orientation carefully.
    raise NotImplementedError


def looping_mdp() -> tuple[FloatArray, FloatArray, FloatArray]:
    """Provided fixture for written Problem 3.

    State 2 is absorbing terminal with zero reward. Tests evaluate only the
    nonterminal values but include the terminal state to exercise shapes.
    """
    transitions = np.zeros((3, 2, 3), dtype=np.float64)
    rewards = np.zeros_like(transitions)

    transitions[0, 0, 1] = 1.0
    transitions[0, 1, 0] = 1.0
    transitions[1, 0, 2] = 1.0
    transitions[1, 1, 0] = 1.0
    transitions[2, :, 2] = 1.0

    rewards[0, 1, 0] = 1.0
    rewards[1, 0, 2] = 3.0
    rewards[1, 1, 0] = -1.0

    policy = np.array([
        [0.60, 0.40],
        [0.25, 0.75],
        [0.50, 0.50],
    ])
    return transitions, rewards, policy
