# CFR Poker Solver: From First Principles to NLHE

A free, open-source course building a Counterfactual Regret Minimization solver from scratch — step by step from Rock-Paper-Scissors to No-Limit Hold'em.

## Live Site

This course is hosted on GitHub Pages. Visit:

**https://rexjensen.github.io/Solver_tutorial/**

## Curriculum

| Chapter | Topic | Game | Key Concepts |
|---------|-------|------|-------------|
| [1](lessons/01-regret-matching.html) | Regret Matching & Nash Equilibrium | Rock-Paper-Scissors | Nash Equilibrium, Regret Matching, Exploitability |
| 2 | *coming soon* | Kuhn Poker | CFR, Game Trees, Information Sets |
| 3 | *coming soon* | Leduc Poker | Multi-street CFR, Betting Rounds |
| 4 | *coming soon* | NLHE Abstractions | Card/Action Abstraction, Bucketing |
| 5 | *coming soon* | NLHE Solver | CFR+ / MCCFR for Real Poker |

## Running the Code

Each chapter has a companion Python script in the `code/` directory:

```bash
pip install numpy
python code/lesson_01_rps.py
```

## Project Structure

```
├── index.html                        # Landing page
├── lessons/
│   └── 01-regret-matching.html       # Chapter 1 (interactive)
├── code/
│   └── lesson_01_rps.py              # Chapter 1 companion code
└── README.md
```

## Deploying with GitHub Pages

1. Go to **Settings > Pages** in your GitHub repository
2. Under **Source**, select **Deploy from a branch**
3. Set the branch to `main` and folder to `/ (root)`
4. Click **Save** — your site will be live within a minute
