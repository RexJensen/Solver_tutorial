# CLAUDE.md — Project Context for Claude Code

## Project Overview

This is an interactive educational course on Counterfactual Regret Minimization (CFR) for poker, hosted as a GitHub Pages site. It progresses from simple games (Rock-Paper-Scissors) through toy poker variants (Kuhn, Leduc) toward No-Limit Hold'em.

## Structure

- `index.html` — Landing page with chapter cards
- `lessons/*.html` — Self-contained interactive HTML lessons (inline CSS + JS, no frameworks)
- `code/*.py` — Companion Python solvers (only dependency: numpy)

## Conventions

- Each chapter has one HTML lesson and one Python solver
- HTML lessons are fully self-contained — no external CSS/JS dependencies
- Interactive demos use vanilla JavaScript with pre-computed Nash strategies
- Python solvers print convergence data and strategy tables to stdout
- Navigation links between chapters are maintained in each lesson's header/footer

## Current State

- Chapters 1-3 are complete
- Chapters 4-5 (NLHE abstractions and solver) are planned but not started

## Key Technical Details

- CFR implementation uses vanilla CFR with regret matching
- Leduc solver handles chance nodes (community card) with proper reach probability propagation
- Best response computation uses a two-pass approach for correctness and efficiency
- Exploitability = sum of best response values for both players (should converge toward 0)
