"""Lecture 4 mastery: value estimation and actor–critic updates."""

from __future__ import annotations

import torch
import torch.nn as nn


def state_values(
    policy_probabilities: torch.Tensor,
    q_values: torch.Tensor,
) -> torch.Tensor:
    """Compute V(s)=sum_a pi(a|s)Q(s,a) for batches shaped (B, A)."""
    # TODO(L4-values): validate distributions and reduce over actions.
    raise NotImplementedError


def advantages_from_q(
    policy_probabilities: torch.Tensor,
    q_values: torch.Tensor,
) -> torch.Tensor:
    """Return A(s,a) with shape (B, A)."""
    # TODO(L4-advantage): reuse state_values without accidental broadcasting.
    raise NotImplementedError


def td_residuals(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Compute delta_t=r_t+gamma*(1-d_t)V(s_{t+1})-V(s_t).

    All inputs have shape (T,). Values are allowed to carry gradients; callers
    decide whether to detach the result for an actor update.
    """
    # TODO(L4-td): handle boolean/float done masks and strict shapes.
    raise NotImplementedError


def n_step_targets(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    n: int,
    gamma: float,
) -> torch.Tensor:
    """Compute an n-step target for every transition in concatenated episodes.

    Args:
        rewards: shape (T,), reward on transitions t=0..T-1.
        dones: boolean shape (T,).
        values: shape (T+1,), values for s_0..s_T.
        n: maximum number of rewards before bootstrapping.

    Stop accumulating and omit bootstrap after any terminal transition. Near the
    end of the supplied rollout, bootstrap from the furthest available state.
    Return shape (T,).
    """
    # TODO(L4-n-step): nested loops are acceptable; correctness beats cleverness.
    raise NotImplementedError


def generalized_advantage_estimates(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (advantages, value_targets) using backwards GAE recursion.

    This is an extension connecting Lecture 4's n-step spectrum to later PPO.
    Every input has shape (T,). A done resets the recursion.
    """
    # TODO(L4-gae): delta plus discounted/lambda-weighted next advantage.
    raise NotImplementedError


def actor_critic_losses(
    selected_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    value_predictions: torch.Tensor,
    value_targets: torch.Tensor,
    value_coefficient: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (actor_loss, critic_loss, combined_loss).

    Actor and critic losses are means. Advantages and value targets are fixed
    targets and must not backpropagate into their estimators.
    """
    # TODO(L4-losses): use a descent-compatible actor sign and MSE critic loss.
    raise NotImplementedError


@torch.no_grad()
def off_policy_bellman_target(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    next_q_from_two_target_critics: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Build a clipped-double-Q target from replay transitions.

    Shapes:
        rewards, dones: (B,)
        next_q_from_two_target_critics: (B, 2)

    Return y=r+gamma*(1-done)*min(q1,q2), shape (B,).
    """
    # TODO(L4-off-policy-target): no target gradients, no broadcasting.
    raise NotImplementedError


@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """In-place target <- (1-tau)*target + tau*source for every parameter."""
    # TODO(L4-target-network): validate tau and matching parameter structure.
    raise NotImplementedError
