# CS224R RL Learn

Personal study workspace for Stanford CS224R (Deep Reinforcement Learning).

## Contents

| Path | Purpose |
|---|---|
| `cs224r-study-pack.html` | Written mastery problems for Lectures 1–4 |
| `practice_templates/` | Typed coding drills + unit tests (not graded homework) |
| `hw1_starter_code/` | Course HW1 starter (Flappy Bird imitation learning) |
| `hw2_4_starter_code/` | Course HW2 starter (Q-learning, off-policy, PPO) |

## Practice drills

```bash
python -m unittest practice_templates.tests.test_mastery.TestMDP -v
python -m unittest practice_templates.tests.test_mastery.TestImitation -v
python -m unittest practice_templates.tests.test_mastery.TestPolicyGradient -v
python -m unittest practice_templates.tests.test_mastery.TestActorCritic -v
```

## Note

This repo is for personal learning. Do not publish graded homework solutions.
Course starter code belongs to Stanford CS224R; keep the repository private.
