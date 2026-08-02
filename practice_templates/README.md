# CS224R Lectures 1–4 Mastery Lab

This is an original practice repository aligned to the Spring 2026 Lecture 1–4
slides. It is intentionally separate from Stanford's homework starter code.

Each lecture file contains typed function contracts, shape requirements, TODOs,
and no completed algorithm. The tests specify behavior without prescribing the
implementation.

## Files

| File | Topics |
|---|---|
| mdp_practice.py | terminal-aware returns, trajectory likelihood, exact policy evaluation, Q/advantage, discounted occupancy |
| imitation_practice.py | action chunks, BC, Gaussian mixtures, flow matching, Euler sampling, DAgger aggregation |
| policy_gradient_practice.py | reward-to-go, categorical score functions, REINFORCE, baselines, importance sampling, KL, PPO clipping |
| actor_critic_practice.py | TD residuals, n-step returns, GAE extension, actor/critic losses, off-policy targets, target-network updates |
| tests/test_mastery.py | executable acceptance checks |

## Work order

1. Read the matching written problems in ../cs224r-study-pack.html.
2. Implement one TODO at a time.
3. Run only that lecture's tests:

    python -m unittest practice_templates.tests.test_mastery.TestMDP -v
    python -m unittest practice_templates.tests.test_mastery.TestImitation -v
    python -m unittest practice_templates.tests.test_mastery.TestPolicyGradient -v
    python -m unittest practice_templates.tests.test_mastery.TestActorCritic -v

Run everything:

    python -m unittest discover -s practice_templates/tests -v

Needs NumPy and PyTorch for the imitation / policy-gradient / actor-critic drills.
