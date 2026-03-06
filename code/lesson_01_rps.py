"""
=============================================================================
LESSON 1: Regret Matching & Nash Equilibrium
Game: Rock-Paper-Scissors
=============================================================================

THEORY
------
Every poker solver (PIOsolver, GTO Wizard, etc.) is built on a core idea
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
   CFR applies regret matching at every decision point (information set)
   in a game tree. The average strategy across all iterations converges
   to Nash Equilibrium. This is the core update idea underlying all
   CFR-based solvers — though production solvers like PIOsolver add
   many practical layers: abstraction, sampling, pruning, discounting,
   and engineering optimizations.

=============================================================================
"""

import numpy as np


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

    This is the fundamental building block of CFR-based solvers.

    The algorithm:
    1. Start with some initial strategy
    2. Compute expected values against opponent's strategy
    3. For each action, compute: "how much better would this action have
       been vs what my strategy earned?" — this is the REGRET
    4. Accumulate regrets over many iterations
    5. Set next strategy proportional to positive cumulative regrets
    6. The AVERAGE strategy (not current strategy) converges to Nash
    """

    def __init__(self, name: str = "Player",
                 initial_strategy: np.ndarray = None):
        self.name = name
        # Cumulative regret for each action (the core state of the algorithm)
        self.cumulative_regret = np.zeros(NUM_ACTIONS)
        # Sum of all strategies played (for computing average strategy)
        self.strategy_sum = np.zeros(NUM_ACTIONS)

        # Allow non-uniform initialization to demonstrate real convergence
        if initial_strategy is not None:
            # Seed the regrets so that get_strategy() produces the desired
            # initial strategy. We scale by an arbitrary constant; the
            # normalization in get_strategy() cancels it out.
            self.cumulative_regret = initial_strategy.copy()

    def get_strategy(self) -> np.ndarray:
        """
        Convert cumulative regrets into a strategy via Regret Matching.

        This is THE key function. You'll see it again in every lesson.

        Rule:
        - Take all positive regrets
        - Normalize them to sum to 1.0
        - If no positive regrets exist, play uniformly

        NOTE: This method is a pure computation. It does NOT update
        strategy_sum — that is done explicitly in the training loop
        to keep concerns separated.
        """
        # Clip negative regrets to zero — we only care about positive regret
        positive_regret = np.maximum(self.cumulative_regret, 0)
        total = positive_regret.sum()

        if total > 0:
            return positive_regret / total
        else:
            # No positive regrets — default to uniform random
            return np.ones(NUM_ACTIONS) / NUM_ACTIONS

    def accumulate_strategy(self, strategy: np.ndarray):
        """
        Add the current iteration's strategy to the running sum.
        Called once per iteration in the training loop.
        """
        self.strategy_sum += strategy

    def get_average_strategy(self) -> np.ndarray:
        """
        The AVERAGE strategy across all iterations — THIS is what converges
        to Nash Equilibrium.

        Important distinction:
        - Current strategy (from get_strategy): oscillates, doesn't converge
        - Average strategy: converges to Nash Equilibrium

        Solvers report this average strategy as the "solution."
        """
        total = self.strategy_sum.sum()
        if total > 0:
            return self.strategy_sum / total
        else:
            return np.ones(NUM_ACTIONS) / NUM_ACTIONS

    def update_regrets(self, my_strategy: np.ndarray,
                       opponent_strategy: np.ndarray):
        """
        Compute and accumulate regrets against the opponent's strategy.

        For each action a:
            regret[a] = EV(always playing a) - EV(my_strategy)

        This tells us: "How much better would action a have been compared
        to what my current strategy earned (in expectation)?"

        Takes my_strategy as an explicit argument to avoid recomputing it
        (which was the source of a double-accumulation bug in v1).
        """
        # EV of each pure action against opponent's mixed strategy
        action_evs = PAYOFF @ opponent_strategy

        # EV of our current mixed strategy
        strategy_ev = my_strategy @ action_evs

        # Regret = (what action would have earned) - (what we earned)
        regrets = action_evs - strategy_ev

        # Accumulate
        self.cumulative_regret += regrets


def compute_exploitability(strategy: np.ndarray) -> float:
    """
    Compute the exploitability of a strategy in a two-player zero-sum game.

    In a two-player zero-sum normal-form game, exploitability of a strategy
    sigma is defined as the best-response value against sigma:

        exploitability(sigma) = max_a [ EV(a, sigma) ]

    where a ranges over the opponent's pure actions.

    At Nash Equilibrium, exploitability = 0, meaning no opponent action
    can achieve positive EV.

    NOTE: In extensive-form games (like poker), exploitability is typically
    defined as the sum of both players' best-response values divided by 2
    (sometimes called "NashConv / 2"). For a symmetric zero-sum game like
    RPS with a symmetric strategy, the single-player version suffices.
    """
    action_evs = PAYOFF @ strategy
    return action_evs.max()


def train_rps(num_iterations: int = 100_000,
              p1_init: np.ndarray = None,
              p2_init: np.ndarray = None,
              verbose: bool = True) -> tuple:
    """
    Train two regret-matching players against each other in RPS.

    By default, both players start with biased (non-equilibrium) strategies
    so that the demo genuinely shows convergence TOWARD equilibrium, rather
    than equilibrium stability from a symmetric initialization.

    If both players start uniform (which is already Nash in RPS), regrets
    stay at zero and nothing "learns." Starting from biased strategies
    makes the convergence visible and honest.
    """
    # Default: start from biased, non-equilibrium strategies
    if p1_init is None:
        p1_init = np.array([0.8, 0.1, 0.1])  # Player 1 favors Rock
    if p2_init is None:
        p2_init = np.array([0.1, 0.1, 0.8])  # Player 2 favors Scissors

    player1 = RegretMatchingPlayer("Player 1", initial_strategy=p1_init)
    player2 = RegretMatchingPlayer("Player 2", initial_strategy=p2_init)

    for i in range(num_iterations):
        # Each player computes their current strategy from regrets
        strategy1 = player1.get_strategy()
        strategy2 = player2.get_strategy()

        # Accumulate strategies ONCE per iteration (separated from get_strategy)
        player1.accumulate_strategy(strategy1)
        player2.accumulate_strategy(strategy2)

        # Each player updates regrets based on opponent's strategy
        player1.update_regrets(strategy1, strategy2)
        player2.update_regrets(strategy2, strategy1)

        # Print progress at exponential intervals
        if verbose and (i + 1) in {1, 10, 100, 1000, 10000, 100000}:
            avg1 = player1.get_average_strategy()
            avg2 = player2.get_average_strategy()
            expl1 = compute_exploitability(avg1)
            expl2 = compute_exploitability(avg2)
            print(f"\n── Iteration {i+1:,} ──")
            print(f"  {player1.name} avg strategy: "
                  f"R={avg1[0]:.4f}  P={avg1[1]:.4f}  S={avg1[2]:.4f}"
                  f"  (exploitability: {expl1:.6f})")
            print(f"  {player2.name} avg strategy: "
                  f"R={avg2[0]:.4f}  P={avg2[1]:.4f}  S={avg2[2]:.4f}"
                  f"  (exploitability: {expl2:.6f})")

    return player1, player2


def train_against_fixed_opponent(
    opponent_strategy: np.ndarray,
    num_iterations: int = 100_000,
    verbose: bool = True,
) -> RegretMatchingPlayer:
    """
    Train a regret-matching player against a FIXED (non-adaptive) opponent.

    This shows how regret matching finds the BEST RESPONSE to any fixed
    opponent strategy. The best response is the action(s) with highest EV.

    KEY INSIGHT FOR POKER:
    Against a non-equilibrium opponent, the best response is NOT Nash.
    Nash is only optimal when the opponent is also playing Nash.
    This is why "exploitative play" exists — it deviates from GTO
    to exploit specific leaks.
    """
    learner = RegretMatchingPlayer("Learner")

    for i in range(num_iterations):
        strategy = learner.get_strategy()
        learner.accumulate_strategy(strategy)
        learner.update_regrets(strategy, opponent_strategy)

        if verbose and (i + 1) in {10, 100, 1000, 10000, 100000}:
            avg = learner.get_average_strategy()
            print(f"\n── Iteration {i+1:,} ──")
            print(f"  Learner avg strategy: "
                  f"R={avg[0]:.4f}  P={avg[1]:.4f}  S={avg[2]:.4f}")

    return learner


def show_best_response_math(opponent_strategy: np.ndarray, label: str):
    """Show the EV calculation for each action against an opponent."""
    print(f"\n  Why? EV of each action vs {label}:")
    action_evs = PAYOFF @ opponent_strategy
    for a in range(NUM_ACTIONS):
        print(f"    EV({ACTION_NAMES[a]:8s}) = "
              f"{' + '.join(f'({PAYOFF[a,o]:+.0f})({opponent_strategy[o]:.1f})' for o in range(NUM_ACTIONS))}"
              f" = {action_evs[a]:+.2f}")
    best = ACTION_NAMES[action_evs.argmax()]
    print(f"  Best response: {best} (EV = {action_evs.max():+.2f})")


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("LESSON 1: Regret Matching in Rock-Paper-Scissors")
    print("=" * 65)

    # ── Demo 1: Self-play convergence to Nash ──
    print("\n\nDEMO 1: Two biased players converging to Nash Equilibrium")
    print("-" * 65)
    print("Player 1 starts biased toward Rock [0.8, 0.1, 0.1].")
    print("Player 2 starts biased toward Scissors [0.1, 0.1, 0.8].")
    print("Watch them converge to [1/3, 1/3, 1/3] and exploitability -> 0.")
    player1, player2 = train_rps(num_iterations=100_000)

    # ── Demo 2: Exploiting a fixed opponent (pure Rock) ──
    print("\n\nDEMO 2: Exploiting an opponent who always plays Rock")
    print("-" * 65)
    always_rock = np.array([1.0, 0.0, 0.0])
    learner = train_against_fixed_opponent(always_rock, num_iterations=100_000)
    show_best_response_math(always_rock, "[R=1.0, P=0.0, S=0.0]")

    # ── Demo 3: Exploiting a biased opponent ──
    print("\n\nDEMO 3: Exploiting opponent [R=0.6, P=0.3, S=0.1]")
    print("-" * 65)
    biased = np.array([0.6, 0.3, 0.1])
    learner = train_against_fixed_opponent(biased, num_iterations=100_000)
    show_best_response_math(biased, "[R=0.6, P=0.3, S=0.1]")

    # ── Demo 4: Exploitability ──
    print("\n\n" + "=" * 65)
    print("DEMO 4: Exploitability as a Convergence Metric")
    print("=" * 65)
    print("""
In a two-player zero-sum game, the exploitability of strategy sigma is:

    exploitability(sigma) = max_a [ EV(a, sigma) ]

This is the best-response EV an opponent can achieve against sigma.
At Nash Equilibrium, exploitability = 0 (no action beats the strategy).

For extensive-form games like poker, solvers typically report exploitability
as (sum of both players' best-response values) / 2, often as a percentage
of the pot. When PIOsolver says "0.3% of pot," that's this metric.
""")
    player1, player2 = train_rps(num_iterations=100_000, verbose=False)
    avg1 = player1.get_average_strategy()
    expl = compute_exploitability(avg1)
    print(f"Player 1 avg strategy: R={avg1[0]:.4f}  P={avg1[1]:.4f}  S={avg1[2]:.4f}")
    print(f"Exploitability: {expl:.6f}  (should be ~0.0 at Nash)")

    # ── Summary ──
    print("\n\n" + "=" * 65)
    print("KEY TAKEAWAYS")
    print("=" * 65)
    print("""
    1. REGRET = "How much better would action X have been?"
    2. REGRET MATCHING = Play actions proportional to positive cumulative regret
    3. The AVERAGE strategy converges to Nash Equilibrium
    4. Against fixed opponents, regret matching finds the BEST RESPONSE
       (the action with highest EV — verify with the math!)
    5. EXPLOITABILITY = max EV an opponent can gain by best-responding.
       At Nash, exploitability = 0. Solvers use this to measure convergence.

    WHAT'S NEXT (Lesson 2):
    We'll apply this same regret matching at every decision point in a
    game TREE — that's CFR (Counterfactual Regret Minimization). We'll
    implement it for Kuhn Poker, a simplified 3-card poker game.
    """)
