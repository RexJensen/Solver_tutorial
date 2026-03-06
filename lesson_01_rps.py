"""
=============================================================================
LESSON 1: Regret Matching & Nash Equilibrium
Game: Rock-Paper-Scissors
=============================================================================

THEORY
------
Every poker solver (PIOsolver, GTO Wizard, etc.) is built on a single idea
called REGRET MATCHING. Before we can understand CFR in poker, we need to
understand regret matching in isolation.

KEY DEFINITIONS:

1. **Strategy**: A probability distribution over actions.
   In RPS: [P(Rock), P(Paper), P(Scissors)]
   Example: [0.4, 0.3, 0.3] means play Rock 40%, Paper 30%, Scissors 30%.

2. **Regret**: How much better you WOULD have done by playing a different
   action, compared to what you actually did (in expectation).

   If your strategy lost 0.2 utils on average, but playing Rock would have
   won 0.5 utils, your regret for Rock = 0.5 - (-0.2) = 0.7

3. **Regret Matching**: Choose your next strategy proportional to your
   CUMULATIVE POSITIVE regrets.

   If cumulative regrets are [3.0, 1.0, 0.0]:
   - Normalize positive regrets: total = 3.0 + 1.0 + 0.0 = 4.0
   - Strategy = [3/4, 1/4, 0] = [0.75, 0.25, 0.0]

   If all regrets are <= 0, play uniformly: [1/3, 1/3, 1/3]

4. **Nash Equilibrium**: A strategy profile where NO player can improve
   their expected value by unilaterally changing strategy.

   In RPS, the Nash Equilibrium is [1/3, 1/3, 1/3].
   Against this strategy, every action yields EV = 0.

WHY THIS MATTERS FOR POKER:
   CFR applies regret matching at EVERY decision point (information set)
   in the game tree. The average strategy across all iterations converges
   to Nash Equilibrium. That's literally what PIOsolver does — it just
   does it across millions of information sets simultaneously.

=============================================================================
"""

import numpy as np
from typing import Optional


# ── Game Definition ────────────────────────────────────────────────────────
# Actions
ROCK, PAPER, SCISSORS = 0, 1, 2
NUM_ACTIONS = 3
ACTION_NAMES = ["Rock", "Paper", "Scissors"]

# Payoff matrix: PAYOFF[my_action][opponent_action] = my utility
# Rock beats Scissors, Scissors beats Paper, Paper beats Rock
PAYOFF = np.array([
    [ 0, -1,  1],   # Rock vs [R, P, S]
    [ 1,  0, -1],   # Paper vs [R, P, S]
    [-1,  1,  0],   # Scissors vs [R, P, S]
], dtype=float)


class RegretMatchingPlayer:
    """
    A player that uses Regret Matching to learn an optimal strategy.

    This is the fundamental building block of ALL CFR-based solvers.
    Everything in PIOsolver/GTO Wizard builds on this exact mechanism.

    The algorithm:
    1. Start with uniform strategy
    2. Play against opponent, observe outcome
    3. For each action you DIDN'T take, compute: "how much better
       would I have done?" — this is the REGRET for that action
    4. Accumulate regrets over many iterations
    5. Set your strategy proportional to positive cumulative regrets
    6. The AVERAGE strategy (not current strategy) converges to Nash
    """

    def __init__(self, name: str = "Player"):
        self.name = name
        # Cumulative regret for each action (the core state of the algorithm)
        self.cumulative_regret = np.zeros(NUM_ACTIONS)
        # Sum of all strategies played (for computing average strategy)
        self.strategy_sum = np.zeros(NUM_ACTIONS)

    def get_strategy(self) -> np.ndarray:
        """
        Convert cumulative regrets into a strategy via Regret Matching.

        This is THE key function. Memorize it. You'll see it again in
        every single lesson going forward.

        Rule:
        - Take all positive regrets
        - Normalize them to sum to 1.0
        - If no positive regrets exist, play uniformly
        """
        # Clip negative regrets to zero — we only care about positive regret
        positive_regret = np.maximum(self.cumulative_regret, 0)
        total = positive_regret.sum()

        if total > 0:
            strategy = positive_regret / total
        else:
            # No positive regrets — default to uniform random
            strategy = np.ones(NUM_ACTIONS) / NUM_ACTIONS

        # Accumulate strategy for computing the average later
        self.strategy_sum += strategy
        return strategy

    def get_average_strategy(self) -> np.ndarray:
        """
        The AVERAGE strategy across all iterations — THIS is what converges
        to Nash Equilibrium.

        Important distinction:
        - Current strategy (from get_strategy): oscillates, doesn't converge
        - Average strategy: converges to Nash Equilibrium

        This is why PIOsolver reports an "average strategy" — it's the
        average over all CFR iterations, not the final iteration's strategy.
        """
        total = self.strategy_sum.sum()
        if total > 0:
            return self.strategy_sum / total
        else:
            return np.ones(NUM_ACTIONS) / NUM_ACTIONS

    def update_regrets(self, opponent_strategy: np.ndarray):
        """
        Compute and accumulate regrets against the opponent's strategy.

        For each action a:
            regret[a] = EV(always playing a) - EV(our current strategy)

        This tells us: "How much better would action a have been compared
        to what we actually did (in expectation)?"
        """
        strategy = self.get_strategy()

        # Expected value of our current mixed strategy
        # For each of our actions, compute EV against opponent's strategy
        action_evs = PAYOFF @ opponent_strategy  # EV of each pure action

        # EV of our current mixed strategy
        strategy_ev = strategy @ action_evs

        # Regret for each action = (what it would have earned) - (what we earned)
        regrets = action_evs - strategy_ev

        # Accumulate regrets
        self.cumulative_regret += regrets


def train_rps(num_iterations: int = 100_000, verbose: bool = True) -> tuple:
    """
    Train two regret-matching players against each other in RPS.

    Both players simultaneously learn. After enough iterations,
    both converge to the Nash Equilibrium: [1/3, 1/3, 1/3].
    """
    player1 = RegretMatchingPlayer("Player 1")
    player2 = RegretMatchingPlayer("Player 2")

    for i in range(num_iterations):
        # Each player computes their current strategy from regrets
        strategy1 = player1.get_strategy()
        strategy2 = player2.get_strategy()

        # Each player updates regrets based on opponent's strategy
        player1.update_regrets(strategy2)
        player2.update_regrets(strategy1)

        # Print progress at exponential intervals
        if verbose and (i + 1) in {10, 100, 1000, 10000, 100000}:
            avg1 = player1.get_average_strategy()
            avg2 = player2.get_average_strategy()
            print(f"\n── Iteration {i+1:,} ──")
            print(f"  {player1.name} avg strategy: "
                  f"R={avg1[0]:.4f}  P={avg1[1]:.4f}  S={avg1[2]:.4f}")
            print(f"  {player2.name} avg strategy: "
                  f"R={avg2[0]:.4f}  P={avg2[1]:.4f}  S={avg2[2]:.4f}")

    return player1, player2


def train_against_fixed_opponent(
    opponent_strategy: np.ndarray,
    num_iterations: int = 100_000,
    verbose: bool = True,
) -> RegretMatchingPlayer:
    """
    Train a regret-matching player against a FIXED (non-adaptive) opponent.

    This is instructive because it shows how regret matching EXPLOITS
    a suboptimal opponent.

    If opponent always plays Rock: learner converges to always Paper.
    If opponent plays Rock 50%, Paper 50%: learner exploits with mostly Paper.

    KEY INSIGHT FOR POKER:
    Against a non-equilibrium opponent, the best response is NOT Nash.
    Nash is only optimal when the opponent is also playing Nash.
    This is why "exploitative play" exists — it deviates from GTO
    to exploit specific leaks.
    """
    learner = RegretMatchingPlayer("Learner")

    for i in range(num_iterations):
        learner.get_strategy()  # updates strategy_sum
        learner.update_regrets(opponent_strategy)

        if verbose and (i + 1) in {10, 100, 1000, 10000, 100000}:
            avg = learner.get_average_strategy()
            print(f"\n── Iteration {i+1:,} ──")
            print(f"  Learner avg strategy: "
                  f"R={avg[0]:.4f}  P={avg[1]:.4f}  S={avg[2]:.4f}")

    return learner


# ── Exercises ──────────────────────────────────────────────────────────────

def exercise_exploitability():
    """
    EXERCISE: Understanding Exploitability

    Exploitability = how much EV an opponent can gain by best-responding
    to your strategy. At Nash Equilibrium, exploitability = 0.

    This is THE metric that PIOsolver uses to determine convergence.
    When it says "converged to 0.3% pot" — that means the exploitability
    is 0.3% of the pot.
    """
    print("=" * 65)
    print("EXERCISE: Computing Exploitability")
    print("=" * 65)

    player1, player2 = train_rps(num_iterations=100_000, verbose=False)

    avg1 = player1.get_average_strategy()
    avg2 = player2.get_average_strategy()

    # Exploitability of player 1 = best response EV against player 1's strategy
    # Best response: for each of opponent's possible actions, pick the one
    # with highest EV against player1's strategy
    action_evs_vs_p1 = PAYOFF @ avg1  # EV of each action vs player1
    best_response_ev = action_evs_vs_p1.max()

    print(f"\nPlayer 1 avg strategy: R={avg1[0]:.4f}  P={avg1[1]:.4f}  S={avg1[2]:.4f}")
    print(f"Player 2 avg strategy: R={avg2[0]:.4f}  P={avg2[1]:.4f}  S={avg2[2]:.4f}")
    print(f"\nBest response EV against Player 1: {best_response_ev:.6f}")
    print(f"(At Nash Equilibrium, this should be ≈ 0.0)")
    print(f"\nExploitability of Player 1: {best_response_ev:.6f}")


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("LESSON 1: Regret Matching in Rock-Paper-Scissors")
    print("=" * 65)

    # ── Demo 1: Self-play convergence to Nash ──
    print("\n\n📊 DEMO 1: Two adaptive players converging to Nash Equilibrium")
    print("-" * 65)
    print("Both players use regret matching. Watch them converge to [1/3, 1/3, 1/3].")
    player1, player2 = train_rps(num_iterations=100_000)

    # ── Demo 2: Exploiting a fixed opponent ──
    print("\n\n📊 DEMO 2: Exploiting an opponent who always plays Rock")
    print("-" * 65)
    print("Against Rock-only, regret matching converges to always Paper.")
    always_rock = np.array([1.0, 0.0, 0.0])
    learner = train_against_fixed_opponent(always_rock, num_iterations=100_000)

    print("\n\n📊 DEMO 3: Exploiting an opponent who plays Rock 60%, Paper 30%, Scissors 10%")
    print("-" * 65)
    biased = np.array([0.6, 0.3, 0.1])
    learner = train_against_fixed_opponent(biased, num_iterations=100_000)

    # ── Demo 4: Exploitability ──
    print("\n")
    exercise_exploitability()

    # ── Summary ──
    print("\n\n" + "=" * 65)
    print("KEY TAKEAWAYS")
    print("=" * 65)
    print("""
    1. REGRET = "How much better would action X have been?"
    2. REGRET MATCHING = Play actions proportional to positive cumulative regret
    3. The AVERAGE strategy converges to Nash Equilibrium
    4. Against fixed opponents, regret matching finds the BEST RESPONSE
    5. EXPLOITABILITY measures distance from Nash — this is how solvers
       measure convergence (e.g., "0.3% of pot")

    WHAT'S NEXT (Lesson 2):
    We'll apply this same regret matching at every decision point in a
    game TREE — that's CFR (Counterfactual Regret Minimization). We'll
    implement it for Kuhn Poker, a simplified 3-card poker game.
    """)
