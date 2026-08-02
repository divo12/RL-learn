"""Executable contracts for the CS224R mastery templates.

The suite is expected to fail until TODOs are implemented.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import torch
import torch.nn as nn

from practice_templates import actor_critic_practice as ac
from practice_templates import imitation_practice as il
from practice_templates import mdp_practice as mdp
from practice_templates import policy_gradient_practice as pg


class TestMDP(unittest.TestCase):
    def test_discounted_returns_respect_episode_boundaries(self):
        actual = mdp.discounted_returns(
            np.array([1.0, 2.0, 10.0, 4.0]),
            np.array([False, True, False, True]),
            gamma=0.5,
        )
        np.testing.assert_allclose(actual, [2.0, 2.0, 12.0, 4.0])

    def test_trajectory_log_probability(self):
        actual = mdp.trajectory_log_probability(
            0.5,
            np.array([0.8, 0.4]),
            np.array([0.3, 0.9]),
        )
        self.assertAlmostEqual(actual, math.log(0.5 * 0.8 * 0.3 * 0.4 * 0.9))

    def test_exact_policy_evaluation_and_advantage_identity(self):
        transitions, rewards, policy = mdp.looping_mdp()
        values = mdp.evaluate_policy(transitions, rewards, policy, gamma=0.9)
        expected = np.linalg.solve(
            np.eye(3) - 0.9 * np.array([
                [0.4, 0.6, 0.0],
                [0.75, 0.0, 0.25],
                [0.0, 0.0, 1.0],
            ]),
            np.array([0.4, 0.0, 0.0]),
        )
        np.testing.assert_allclose(values, expected)

        q_values = mdp.action_values(transitions, rewards, values, gamma=0.9)
        advantage = mdp.advantages(policy, q_values)
        np.testing.assert_allclose((policy * advantage).sum(axis=1), 0.0, atol=1e-10)

    def test_discounted_occupancy_is_a_distribution(self):
        transitions, _, policy = mdp.looping_mdp()
        occupancy = mdp.discounted_state_occupancy(
            transitions,
            policy,
            np.array([1.0, 0.0, 0.0]),
            gamma=0.9,
        )
        self.assertAlmostEqual(float(occupancy.sum()), 1.0)
        self.assertTrue(np.all(occupancy >= 0.0))


class TestImitation(unittest.TestCase):
    def test_action_chunks_do_not_pad(self):
        states = np.arange(10, dtype=np.float32).reshape(5, 2)
        actions = np.arange(5, dtype=np.float32).reshape(5, 1)
        chunk_states, chunks = il.make_action_chunks(states, actions, horizon=3)
        np.testing.assert_array_equal(chunk_states, states[:3])
        np.testing.assert_array_equal(
            chunks[:, :, 0],
            np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]], dtype=np.float32),
        )

    def test_chunk_mlp_shape_bounds_and_gradient(self):
        policy = il.ChunkMLP(4, 2, horizon=3, hidden_dims=(16, 8))
        states = torch.randn(5, 4)
        output = policy(states)
        self.assertEqual(tuple(output.shape), (5, 3, 2))
        self.assertTrue(torch.all((0.0 <= output) & (output <= 1.0)))
        output.mean().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in policy.parameters()))

    def test_bc_loss_rejects_broadcasting_by_matching_exact_shape(self):
        policy = nn.Identity()
        states = torch.tensor([[0.0, 1.0], [1.0, 1.0]])
        targets = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
        actual = il.behavior_cloning_mse(policy, states, targets)
        self.assertAlmostEqual(float(actual), 0.5)

    def test_one_component_gmm_is_standard_normal(self):
        logits = torch.zeros(2, 1)
        means = torch.zeros(2, 1, 1)
        log_stds = torch.zeros(2, 1, 1)
        targets = torch.zeros(2, 1)
        actual = il.diagonal_gmm_nll(logits, means, log_stds, targets)
        self.assertAlmostEqual(float(actual), 0.5 * math.log(2.0 * math.pi), places=6)

    def test_flow_interpolation(self):
        clean = torch.tensor([[0.2, 0.8]])
        noise = torch.tensor([[-1.0, 1.0]])
        sample, velocity = il.flow_interpolate(clean, torch.tensor([0.25]), noise)
        torch.testing.assert_close(sample, torch.tensor([[-0.7, 0.95]]))
        torch.testing.assert_close(velocity, torch.tensor([[1.2, -0.2]]))

    def test_euler_constant_vector_field(self):
        def field(actions, states, tau):
            del states, tau
            return torch.ones_like(actions)

        actual = il.euler_sample(
            field,
            states=torch.zeros(2, 3),
            initial_noise=torch.zeros(2, 4),
            num_steps=4,
        )
        torch.testing.assert_close(actual, torch.ones(2, 4))

    def test_dagger_relabeling_keeps_episode_boundaries(self):
        episodes = [
            np.arange(4, dtype=np.float32).reshape(4, 1),
            np.arange(10, 13, dtype=np.float32).reshape(3, 1),
        ]

        def expert(state):
            return state[0] * 2.0

        states, chunks = il.relabel_episodes(episodes, expert, horizon=2)
        self.assertEqual(states.shape, (5, 1))
        self.assertEqual(chunks.shape, (5, 2, 1))
        np.testing.assert_array_equal(chunks[2, :, 0], [4.0, 6.0])
        np.testing.assert_array_equal(chunks[3, :, 0], [20.0, 22.0])


class TestPolicyGradient(unittest.TestCase):
    def test_reward_to_go(self):
        rewards = torch.tensor([2.0, -1.0, 0.0, 5.0])
        dones = torch.tensor([False, True, False, True])
        actual = pg.reward_to_go(rewards, dones, gamma=0.9)
        torch.testing.assert_close(actual, torch.tensor([1.1, -1.0, 4.5, 5.0]))

    def test_categorical_score_vectors(self):
        logits = torch.tensor([[0.0, 0.0], [math.log(3.0), 0.0]])
        actions = torch.tensor([0, 1])
        actual = pg.categorical_score_vectors(logits, actions)
        expected = torch.tensor([[0.5, -0.5], [-0.75, 0.75]])
        torch.testing.assert_close(actual, expected)

    def test_reinforce_descent_sign_and_detached_weights(self):
        log_probs = torch.tensor([-1.0, -2.0], requires_grad=True)
        returns = torch.tensor([3.0, 1.0], requires_grad=True)
        baseline = torch.tensor([1.0, 1.0], requires_grad=True)
        loss = pg.reinforce_loss(log_probs, returns, baseline)
        self.assertAlmostEqual(float(loss), 1.0)
        loss.backward()
        self.assertIsNotNone(log_probs.grad)
        self.assertIsNone(returns.grad)
        self.assertIsNone(baseline.grad)

    def test_variance_optimal_constant_baseline(self):
        scores = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
        returns = torch.tensor([1.0, 3.0])
        actual = pg.optimal_constant_baseline(scores, returns)
        self.assertAlmostEqual(float(actual), 13.0 / 5.0)

    def test_importance_ratio_and_effective_sample_size(self):
        target = torch.log(torch.tensor([0.4, 0.5, 0.6]))
        behavior = torch.log(torch.tensor([0.8, 0.4, 0.2]))
        per_step, trajectory = pg.trajectory_importance_ratio(target, behavior)
        torch.testing.assert_close(per_step, torch.tensor([0.5, 1.25, 3.0]))
        self.assertAlmostEqual(float(trajectory), 1.875)

        weights = torch.tensor([0.1, 0.2, 0.7, 4.0])
        self.assertAlmostEqual(float(pg.effective_sample_size(weights)), 25.0 / 16.54, places=6)

    def test_categorical_kl_and_ppo_clipping(self):
        logits = torch.zeros(3, 2)
        self.assertAlmostEqual(float(pg.categorical_kl(logits, logits)), 0.0)

        old = torch.zeros(2)
        new = torch.log(torch.tensor([1.4, 0.5]))
        advantages = torch.tensor([3.0, -3.0])
        loss = pg.ppo_clipped_loss(new, old, advantages, clip_epsilon=0.2)
        expected_surrogates = torch.tensor([1.2 * 3.0, 0.5 * -3.0])
        self.assertAlmostEqual(float(loss), float(-expected_surrogates.mean()))


class TestActorCritic(unittest.TestCase):
    def test_values_and_advantage_identity(self):
        probabilities = torch.tensor([[0.2, 0.3, 0.5]])
        q_values = torch.tensor([[1.0, 4.0, 7.0]])
        values = ac.state_values(probabilities, q_values)
        advantage = ac.advantages_from_q(probabilities, q_values)
        torch.testing.assert_close(values, torch.tensor([4.9]))
        torch.testing.assert_close((probabilities * advantage).sum(-1), torch.zeros(1))

    def test_td_and_n_step_targets(self):
        rewards = torch.tensor([2.0, -1.0, 0.0, 5.0])
        dones = torch.tensor([False, False, False, True])
        values = torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0])

        delta = ac.td_residuals(
            rewards,
            dones,
            values[:-1],
            values[1:],
            gamma=0.9,
        )
        torch.testing.assert_close(delta, torch.tensor([0.7, -2.2, 0.9, 4.0]))

        targets = ac.n_step_targets(rewards, dones, values, n=2, gamma=0.9)
        torch.testing.assert_close(targets, torch.tensor([2.72, -0.19, 4.5, 5.0]))

    def test_gae_lambda_zero_matches_one_step_td(self):
        rewards = torch.tensor([2.0, 5.0])
        dones = torch.tensor([False, True])
        values = torch.tensor([4.0, 1.0])
        next_values = torch.tensor([1.0, 0.0])
        advantage, targets = ac.generalized_advantage_estimates(
            rewards, dones, values, next_values, gamma=0.9, gae_lambda=0.0
        )
        torch.testing.assert_close(advantage, torch.tensor([-1.1, 4.0]))
        torch.testing.assert_close(targets, torch.tensor([2.9, 5.0]))

    def test_actor_critic_losses_detach_targets(self):
        log_probs = torch.tensor([-0.2, -1.1, -0.7], requires_grad=True)
        advantage = torch.tensor([1.5, -0.5, 2.0], requires_grad=True)
        predictions = torch.tensor([2.0, 1.0, 4.0], requires_grad=True)
        targets = torch.tensor([3.0, 0.0, 5.0], requires_grad=True)
        actor, critic, total = ac.actor_critic_losses(
            log_probs, advantage, predictions, targets
        )
        self.assertAlmostEqual(float(actor), 1.15 / 3.0, places=6)
        self.assertAlmostEqual(float(critic), 1.0)
        total.backward()
        self.assertIsNotNone(log_probs.grad)
        self.assertIsNotNone(predictions.grad)
        self.assertIsNone(advantage.grad)
        self.assertIsNone(targets.grad)

    def test_off_policy_target_and_soft_update(self):
        actual = ac.off_policy_bellman_target(
            rewards=torch.tensor([1.0, 2.0]),
            dones=torch.tensor([False, True]),
            next_q_from_two_target_critics=torch.tensor([[3.0, 4.0], [8.0, 2.0]]),
            gamma=0.5,
        )
        torch.testing.assert_close(actual, torch.tensor([2.5, 2.0]))

        target = nn.Linear(1, 1, bias=False)
        source = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            target.weight.fill_(0.0)
            source.weight.fill_(2.0)
        ac.soft_update(target, source, tau=0.25)
        torch.testing.assert_close(target.weight, torch.tensor([[0.5]]))


if __name__ == "__main__":
    unittest.main()
