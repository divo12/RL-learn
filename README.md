# CS224R RL Learn

Personal study workspace for Stanford CS224R (Deep Reinforcement Learning), Lectures 1–4.

## Contents

| Path | Purpose |
|---|---|
| `cs224r-study-pack.html` | Written mastery problems for Lectures 1–4 |
| `lecture-notes/` | Revision notes distilled from lectures |
| `Tutor-log/` | Socratic tutor dialogue logs per problem |
| `practice_templates/` | Typed coding drills + unit tests |

## Practice drills

```bash
python -m unittest practice_templates.tests.test_mastery.TestMDP -v
python -m unittest practice_templates.tests.test_mastery.TestImitation -v
python -m unittest practice_templates.tests.test_mastery.TestPolicyGradient -v
python -m unittest practice_templates.tests.test_mastery.TestActorCritic -v
```

Or everything:

```bash
python -m unittest discover -s practice_templates/tests -v
```

These are original practice problems, not course homework solutions.
