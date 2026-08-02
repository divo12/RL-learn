"""Lecture 3 mastery: policy gradients and controlled data reuse."""

from __future__ import annotations

import torch


def reward_to_go(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Discounted reward-to-go for concatenated episodes.

    rewards: shape (T,)
    dones: boolean shape (T,); true means that transition ends an episode.
    """
    # TODO(L3-causality): backwards recursion with terminal resets.
    raise NotImplementedError


def categorical_score_vectors(
    logits: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    """Compute d log pi(a) / d logits analytically.

    Args:
        logits: shape (B, A).
        actions: integer shape (B,).

    Returns:
        score vectors with shape (B, A), equal to one_hot(actions)-softmax.
    """
    # TODO(L3-score): do not use autograd for this analytical exercise.
    raise NotImplementedError


def reinforce_loss(
    selected_log_probs: torch.Tensor,
    returns: torch.Tensor,
    baseline: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return a scalar loss for gradient descent.

    Baseline values are fixed targets for the policy update. Reject shapes that
    would broadcast. The sign must implement gradient ascent on expected return.
    """
    # TODO(L3-reinforce): detach policy weights and average samples.
    raise NotImplementedError


def optimal_constant_baseline(
    score_vectors: torch.Tensor,
    returns: torch.Tensor,
) -> torch.Tensor:
    """Variance-minimizing scalar baseline for score-function samples.

    Minimize mean(||g_i||^2 * (G_i-b)^2). Return a scalar tensor and handle a
    zero denominator explicitly.
    """
    # TODO(L3-baseline): derive the weighted mean rather than using mean(G).
    raise NotImplementedError


def trajectory_importance_ratio(
    target_log_probs: torch.Tensor,
    behavior_log_probs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (per_step_ratios, full_trajectory_ratio) in a stable way.

    Inputs are selected-action log-probabilities for one trajectory, shape (T,).
    Compute the product through a sum in log-space.
    """
    # TODO(L3-importance): validate finite values before exponentiating.
    raise NotImplementedError


def effective_sample_size(weights: torch.Tensor) -> torch.Tensor:
    """Return (sum w)^2 / sum(w^2) for non-negative one-dimensional weights."""
    # TODO(L3-ess): reject negative weights and a zero total.
    raise NotImplementedError


def categorical_kl(
    old_logits: torch.Tensor,
    new_logits: torch.Tensor,
) -> torch.Tensor:
    """Mean D_KL(pi_old || pi_new) for categorical policy batches."""
    # TODO(L3-kl): remain in log-probability space.
    raise NotImplementedError


def ppo_clipped_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float,
) -> torch.Tensor:
    """Negative mean PPO clipped surrogate for gradient descent.

    old_log_probs and advantages are fixed data. All inputs have shape (B,).
    """
    # TODO(L3-ppo): ratio from log probabilities, clipped/unclipped minimum.
    raise NotImplementedError
