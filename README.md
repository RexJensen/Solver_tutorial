# CFR Poker Solver: From First Principles to NLHE

A free, open-source course building a Counterfactual Regret Minimization (CFR) solver from scratch — step by step from Rock-Paper-Scissors to No-Limit Hold'em.

Each chapter includes an **interactive HTML lesson** with visualizations, live demos, and a play-against-the-bot mode, plus a companion **Python solver** you can run locally.

## Live Site

This course is hosted on GitHub Pages. Visit:

**https://rexjensen.github.io/Solver_tutorial/**

## Curriculum

| Chapter | Topic | Game | Key Concepts |
|---------|-------|------|-------------|
| [1](lessons/01-regret-matching.html) | Regret Matching & Nash Equilibrium | Rock-Paper-Scissors | Nash Equilibrium, Regret Matching, Exploitability |
| [2](lessons/02-kuhn-poker.html) | Kuhn Poker & CFR | Kuhn Poker | CFR, Game Trees, Information Sets |
| [3](lessons/03-leduc-poker.html) | Leduc Poker & Multi-Street CFR | Leduc Poker | Chance Nodes, Multi-Street Games, Best Response |
| 4 | *coming soon* | NLHE Abstractions | Card/Action Abstraction, Bucketing |
| 5 | *coming soon* | NLHE Solver | CFR+ / MCCFR for Real Poker |

## Running the Code

Each chapter has a companion Python script in the `code/` directory:

```bash
pip install numpy
python code/lesson_01_rps.py
python code/lesson_02_kuhn.py
python code/lesson_03_leduc.py
```

## Project Structure

```
├── index.html                        # Landing page
├── lessons/
│   ├── 01-regret-matching.html       # Chapter 1: Regret Matching
│   ├── 02-kuhn-poker.html            # Chapter 2: Kuhn Poker & CFR
│   └── 03-leduc-poker.html           # Chapter 3: Leduc Poker & Multi-Street CFR
├── code/
│   ├── lesson_01_rps.py              # RPS regret matching solver
│   ├── lesson_02_kuhn.py             # Kuhn Poker CFR solver
│   └── lesson_03_leduc.py            # Leduc Poker CFR solver
└── README.md
```

## Deploying with GitHub Pages

1. Go to **Settings > Pages** in your GitHub repository
2. Under **Source**, select **Deploy from a branch**
3. Set the branch to `main` and folder to `/ (root)`
4. Click **Save** — your site will be live within a minute

## Built With Claude

This project was built collaboratively with [Claude](https://claude.ai) (Anthropic's AI assistant). Claude wrote the CFR solver implementations, designed the interactive HTML lessons, debugged convergence issues in the multi-street solver, and authored the educational content. The project owner directed the curriculum, reviewed the output, and guided the pedagogical approach.
