"""
=============================================================================
LESSON 2: Kuhn Poker & Counterfactual Regret Minimization
Game: Kuhn Poker (3-card variant)
=============================================================================

THEORY
------
In Lesson 1, we built regret matching for a normal-form game (RPS) — a single
simultaneous decision. Poker is an EXTENSIVE-FORM game: decisions happen
sequentially in a tree, with hidden information (private cards).

CFR (Counterfactual Regret Minimization) extends regret matching to game trees:
    - Place a regret-matching instance at EVERY INFORMATION SET
    - Use COUNTERFACTUAL VALUES instead of expected values
    - Traverse the tree recursively, propagating values upward

KUHN POKER RULES:
    - Deck: {Jack, Queen, King} (3 cards)
    - 2 players each ante 1 chip, dealt 1 card each
    - Player 1 acts first: Check or Bet (1 chip)
    - If Check:  Player 2 can Check (showdown) or Bet (1 chip)
    -   If P2 Bets: Player 1 can Fold (lose ante) or Call (1 chip)
    - If Bet:  Player 2 can Fold (lose ante) or Call (1 chip)
    - At showdown, higher card wins the pot

INFORMATION SETS:
    A player's information set = their card + the betting history.
    E.g., "Q:cb" means "I hold Queen, P1 checked, P2 bet."

    Player 1 info sets: J:, Q:, K:      (initial decision)
                        J:cb, Q:cb, K:cb (facing P2 bet after checking)
    Player 2 info sets: J:c, Q:c, K:c   (facing P1 check)
                        J:b, Q:b, K:b   (facing P1 bet)
    Total: 12 information sets

KNOWN NASH EQUILIBRIUM (parameterized by alpha, 0 <= alpha <= 1/3):
    Player 1:
        K:    bet with probability 3*alpha    (value bet)
        Q:    always check                    (marginal hand)
        J:    bet with probability alpha      (bluff)
        K:cb  always call
        Q:cb  call with probability alpha + 1/3
        J:cb  always fold

    Player 2:
        K:b   always call
        Q:b   call with probability 1/3
        J:b   always fold
        K:c   always bet
        Q:c   always check
        J:c   bet with probability 1/3        (bluff)

    Game value: -1/18 per hand for Player 1 (second player has positional advantage)

CFR ALGORITHM:
    At each information set I with actions A(I):

    1. Compute strategy from regrets (same as Lesson 1):
       sigma(a) = max(R(a), 0) / sum(max(R(a'), 0)) for all a'
       (uniform if all regrets <= 0)

    2. Compute COUNTERFACTUAL VALUE for each action:
       v(I, a) = value of playing action a at I, assuming the player
       "tries to reach" I (reach probability = 1 for own actions leading here)

    3. Update regrets:
       R(a) += v(I, a) - sum_a'[sigma(a') * v(I, a')]

    4. Accumulate weighted strategy:
       strategy_sum(a) += reach_probability * sigma(a)

    The AVERAGE strategy converges to Nash Equilibrium.

=============================================================================
"""

import numpy as np
from collections import defaultdict


# ── Game Definition ────────────────────────────────────────────────────────

# Card values: higher beats lower at showdown
JACK, QUEEN, KING = 0, 1, 2
CARD_NAMES = {0: "J", 1: "Q", 2: "K"}
NUM_ACTIONS = 2  # 0 = Pass (check/fold), 1 = Bet (bet/call)
ACTION_NAMES = ["Pass", "Bet"]


class KuhnCFR:
    """
    Vanilla CFR for Kuhn Poker.

    This is the full CFR algorithm from Zinkevich et al. (2007), applied to
    the simplest non-trivial poker game. The same structure scales to any
    extensive-form game — larger games just have more information sets.

    Key data structures (per information set):
        cumulative_regret[info_set][action] - running sum of counterfactual regrets
        strategy_sum[info_set][action]      - reach-weighted sum of strategies
    """

    def __init__(self):
        # Maps info_set_key -> np.array of size NUM_ACTIONS
        self.cumulative_regret = defaultdict(lambda: np.zeros(NUM_ACTIONS))
        self.strategy_sum = defaultdict(lambda: np.zeros(NUM_ACTIONS))
        self.num_iterations = 0

    def get_strategy(self, info_set: str) -> np.ndarray:
        """
        Convert cumulative regrets to a strategy via regret matching.

        Identical to Lesson 1's get_strategy, but now operates per-info-set.
        This is called once per info set per traversal.
        """
        regret = self.cumulative_regret[info_set]
        positive = np.maximum(regret, 0)
        total = positive.sum()

        if total > 0:
            return positive / total
        else:
            return np.ones(NUM_ACTIONS) / NUM_ACTIONS

    def get_average_strategy(self, info_set: str) -> np.ndarray:
        """
        The average strategy at this info set — this is the Nash Equilibrium
        approximation that we report as the solution.
        """
        s = self.strategy_sum[info_set]
        total = s.sum()
        if total > 0:
            return s / total
        else:
            return np.ones(NUM_ACTIONS) / NUM_ACTIONS

    def cfr(self, cards: list, history: str, reach_0: float, reach_1: float) -> float:
        """
        Recursive CFR traversal of the Kuhn Poker game tree.

        Parameters
        ----------
        cards : list
            cards[0] = Player 0's card, cards[1] = Player 1's card
        history : str
            Betting history so far. Each character is an action:
            'c' = check/fold (pass), 'b' = bet/call
        reach_0 : float
            Player 0's contribution to the reach probability
            (product of Player 0's strategy probs on the path to here)
        reach_1 : float
            Player 1's contribution to the reach probability

        Returns
        -------
        float
            The expected value for Player 0 at this node.
            (Player 1's value is the negative, since this is zero-sum.)

        The reach probabilities are the key concept that separates CFR from
        simple regret matching:
        - reach_i = product of player i's strategy probabilities along the
          path from root to this node
        - The COUNTERFACTUAL value for player i weights outcomes by the
          OPPONENT's reach probability, not the player's own
        - This "what if I tried to reach here?" framing is why it's called
          "counterfactual"
        """
        # ── Terminal node checks ──
        plays = len(history)
        acting_player = plays % 2  # Player 0 acts on even turns, Player 1 on odd

        # Check if the hand is over
        if plays >= 2:
            # Both players have acted at least once
            is_fold = history[-1] == 'c' and history[-2] == 'b'  # bet then fold
            is_showdown_after_checks = history == "cc"  # both checked
            is_showdown_after_call = history[-1] == 'b' and history[-2] == 'b'  # bet then call
            # Also handle: check, bet, call = "cbb"
            if len(history) == 3:
                is_fold = history == "cbc"  # check, bet, fold
                is_showdown_after_call = history == "cbb"  # check, bet, call

            if is_fold:
                # The player who folded loses their ante (and bet if they called then folded)
                # In Kuhn poker, a fold always means the folder loses 1 (the ante)
                # The folding player is the one who just acted (played 'c' after a 'b')
                folder = acting_player  # last action was by (plays-1)%2, but plays already advanced
                # Actually: history[-1] is the last action, taken by player (plays-1)%2
                folder = (plays - 1) % 2
                return 1.0 if folder != 0 else -1.0

            if is_showdown_after_checks:
                # Both checked — showdown for 1 chip (the ante)
                return 1.0 if cards[0] > cards[1] else -1.0

            if is_showdown_after_call:
                # Bet and call — showdown for 2 chips (ante + bet)
                return 2.0 if cards[0] > cards[1] else -2.0

        # ── Non-terminal: compute strategy and recurse ──
        info_set = CARD_NAMES[cards[acting_player]] + ":" + history

        strategy = self.get_strategy(info_set)
        action_values = np.zeros(NUM_ACTIONS)
        node_value = 0.0

        for action in range(NUM_ACTIONS):
            action_char = 'c' if action == 0 else 'b'
            next_history = history + action_char

            if acting_player == 0:
                # Player 0 acts: their reach probability is multiplied by strategy[action]
                action_values[action] = self.cfr(
                    cards, next_history,
                    reach_0 * strategy[action], reach_1
                )
            else:
                # Player 1 acts: their reach probability is multiplied
                action_values[action] = self.cfr(
                    cards, next_history,
                    reach_0, reach_1 * strategy[action]
                )

            node_value += strategy[action] * action_values[action]

        # ── Regret update ──
        # Counterfactual regret is weighted by the OPPONENT's reach probability.
        # This is the "counterfactual" part: we assume the current player
        # TRIES to reach this info set (their own reach = 1), and weight
        # by the probability that the opponent's actions lead here.
        opponent_reach = reach_1 if acting_player == 0 else reach_0
        my_reach = reach_0 if acting_player == 0 else reach_1

        for action in range(NUM_ACTIONS):
            regret = action_values[action] - node_value
            # Note: values are from Player 0's perspective.
            # If acting_player == 1, we need to negate (Player 1 maximizes -EV_0)
            if acting_player == 1:
                regret = -regret
            self.cumulative_regret[info_set][action] += opponent_reach * regret

        # Accumulate strategy weighted by the current player's reach probability
        self.strategy_sum[info_set] += my_reach * strategy

        return node_value

    def train(self, num_iterations: int, verbose: bool = True):
        """
        Run CFR for the specified number of iterations.

        Each iteration deals all possible card combinations (6 permutations)
        and traverses the game tree for each. This is "full traversal" CFR —
        no sampling.
        """
        # All possible deals: (P0_card, P1_card) where cards differ
        deals = [(c0, c1) for c0 in range(3) for c1 in range(3) if c0 != c1]

        for i in range(num_iterations):
            for cards in deals:
                self.cfr(list(cards), "", 1.0, 1.0)
            self.num_iterations += 1

            # Print progress at key checkpoints
            if verbose and (i + 1) in {1, 10, 100, 1000, 10000, 50000, 100000}:
                self._print_progress(i + 1)

        return self

    def _print_progress(self, iteration: int):
        """Print current average strategies and exploitability."""
        print(f"\n── Iteration {iteration:,} ──")
        exploit = self.compute_exploitability()
        print(f"  Exploitability: {exploit:.6f}")

        # Show strategies at all info sets, grouped by player
        for player, label in [(0, "Player 1"), (1, "Player 2")]:
            print(f"  {label}:")
            info_sets = sorted([k for k in self.strategy_sum.keys()
                                if self._info_set_player(k) == player])
            for info_set in info_sets:
                avg = self.get_average_strategy(info_set)
                card = info_set[0]
                hist = info_set[2:] if len(info_set) > 2 else "(opening)"
                print(f"    {info_set:6s}  Pass={avg[0]:.3f}  Bet={avg[1]:.3f}"
                      f"  [{card} after {hist}]")

    def _info_set_player(self, info_set: str) -> int:
        """Determine which player acts at this info set from the history length."""
        history = info_set[2:]  # everything after "X:"
        return len(history) % 2

    def compute_exploitability(self) -> float:
        """
        Compute exploitability of the current average strategy profile.

        Exploitability = (BR_value_P0 + BR_value_P1) / 2
        where BR_value_Pi is the expected value of a best response against
        player i's average strategy.

        IMPORTANT: The best response must respect information sets — the BR
        player must make the SAME decision at all histories belonging to the
        same info set (they can't see the opponent's card). This is computed
        by processing info sets bottom-up and choosing the action that
        maximizes the TOTAL expected value across all consistent deals.
        """
        deals = [(c0, c1) for c0 in range(3) for c1 in range(3) if c0 != c1]
        br_value_0 = self._compute_br_value(deals, br_player=0)
        br_value_1 = self._compute_br_value(deals, br_player=1)
        return max(0.0, (br_value_0 + br_value_1) / 2.0)

    def _compute_br_value(self, deals: list, br_player: int) -> float:
        """
        Compute the best-response value for br_player against the other
        player's average strategy.

        1. Determine the BR strategy by processing info sets bottom-up
        2. Evaluate the expected game value with the BR strategy
        """
        # Step 1: Compute BR strategy bottom-up (deepest info sets first)
        info_sets_by_depth = defaultdict(list)
        for k in self.strategy_sum.keys():
            if self._info_set_player(k) == br_player:
                depth = len(k) - 2  # history length
                info_sets_by_depth[depth].append(k)

        br_strategy = {}
        card_map = {"J": JACK, "Q": QUEEN, "K": KING}

        for depth in sorted(info_sets_by_depth.keys(), reverse=True):
            for info_set in info_sets_by_depth[depth]:
                card_idx = card_map[info_set[0]]
                history = info_set[2:]

                # Sum action values over all consistent deals
                action_evs = np.zeros(NUM_ACTIONS)
                for cards in deals:
                    if cards[br_player] != card_idx:
                        continue
                    for action in range(NUM_ACTIONS):
                        action_char = 'c' if action == 0 else 'b'
                        val = self._eval_tree(
                            cards, history + action_char, br_player, br_strategy)
                        action_evs[action] += val

                # Pick the best action
                best = np.argmax(action_evs)
                s = np.zeros(NUM_ACTIONS)
                s[best] = 1.0
                br_strategy[info_set] = s

        # Step 2: Evaluate overall expected value with BR strategy
        total = 0.0
        for cards in deals:
            total += self._eval_tree(list(cards), "", br_player, br_strategy)
        return total / len(deals)

    def _eval_tree(self, cards: list, history: str, br_player: int,
                   br_strategy: dict) -> float:
        """
        Evaluate the game tree for a single deal, where br_player follows
        br_strategy and the other player follows their average strategy.

        Returns value from br_player's perspective.
        """
        p0_val = self._terminal_value(cards, history)
        if p0_val is not None:
            return p0_val if br_player == 0 else -p0_val

        acting = len(history) % 2
        info_set = CARD_NAMES[cards[acting]] + ":" + history

        if acting == br_player:
            strategy = br_strategy.get(info_set, np.ones(NUM_ACTIONS) / NUM_ACTIONS)
        else:
            strategy = self.get_average_strategy(info_set)

        value = 0.0
        for action in range(NUM_ACTIONS):
            action_char = 'c' if action == 0 else 'b'
            val = self._eval_tree(cards, history + action_char, br_player, br_strategy)
            value += strategy[action] * val
        return value

    def _terminal_value(self, cards: list, history: str) -> float:
        """Return terminal value from Player 0's perspective, or None if not terminal."""
        plays = len(history)

        if plays < 2:
            return None

        if history == "cc":
            return 1.0 if cards[0] > cards[1] else -1.0
        if history == "bc":
            return 1.0  # P1 folds to P0's bet
        if history == "bb":
            return 2.0 if cards[0] > cards[1] else -2.0
        if history == "cbc":
            return -1.0  # P0 folds to P1's bet
        if history == "cbb":
            return 2.0 if cards[0] > cards[1] else -2.0

        return None

    def print_final_strategy(self):
        """Print the final average strategy in a readable format."""
        print("\n" + "=" * 65)
        print("FINAL AVERAGE STRATEGY (Nash Equilibrium Approximation)")
        print("=" * 65)

        for player, label in [(0, "Player 1 (first to act)"), (1, "Player 2")]:
            print(f"\n  {label}:")
            info_sets = sorted([k for k in self.strategy_sum.keys()
                                if self._info_set_player(k) == player])
            for info_set in info_sets:
                avg = self.get_average_strategy(info_set)
                card = info_set[0]
                hist = info_set[2:]
                # Describe the situation
                if player == 0 and hist == "":
                    situation = f"Holding {card}, opening action"
                elif player == 0 and hist == "cb":
                    situation = f"Holding {card}, checked, opponent bet"
                elif player == 1 and hist == "c":
                    situation = f"Holding {card}, opponent checked"
                elif player == 1 and hist == "b":
                    situation = f"Holding {card}, opponent bet"
                else:
                    situation = f"Holding {card}, history={hist}"

                # Format nicely
                pass_pct = avg[0] * 100
                bet_pct = avg[1] * 100
                print(f"    {info_set:6s}  Pass {pass_pct:5.1f}%  |  Bet {bet_pct:5.1f}%"
                      f"    ({situation})")


def compare_with_known_equilibrium(solver: KuhnCFR):
    """
    Compare the learned strategy against the known analytical Nash Equilibrium.

    The Nash Equilibrium for Kuhn Poker is parameterized by alpha in [0, 1/3].
    We infer alpha from the learned strategy (Player 1's bluff frequency with Jack)
    and then compare all info sets against the corresponding equilibrium.
    """
    # Infer alpha from the learned bluff frequency with Jack
    j_strat = solver.get_average_strategy("J:")
    alpha = j_strat[1]  # P(Bet) with Jack = alpha
    alpha = np.clip(alpha, 0, 1/3)  # Clamp to valid range

    print("\n" + "=" * 65)
    print(f"COMPARISON WITH KNOWN NASH EQUILIBRIUM")
    print(f"  (inferred alpha = {alpha:.4f}, valid range [0, 0.3333])")
    print("=" * 65)

    # Known equilibrium strategies: {info_set: [pass_prob, bet_prob]}
    known = {
        # Player 1 opening
        "K:":   [1 - 3*alpha, 3*alpha],       # bet with prob 3*alpha
        "Q:":   [1.0, 0.0],                    # always check
        "J:":   [1 - alpha, alpha],            # bluff with prob alpha
        # Player 1 facing bet after check
        "K:cb": [0.0, 1.0],                    # always call
        "Q:cb": [2/3 - alpha, 1/3 + alpha],    # call with prob 1/3 + alpha
        "J:cb": [1.0, 0.0],                    # always fold
        # Player 2 facing check
        "K:c":  [0.0, 1.0],                    # always bet
        "Q:c":  [1.0, 0.0],                    # always check
        "J:c":  [2/3, 1/3],                    # bluff with prob 1/3
        # Player 2 facing bet
        "K:b":  [0.0, 1.0],                    # always call
        "Q:b":  [2/3, 1/3],                    # call with prob 1/3
        "J:b":  [1.0, 0.0],                    # always fold
    }

    print(f"\n  {'Info Set':8s}  {'Action':6s}  {'Learned':>8s}  {'Known':>8s}  {'Error':>8s}")
    print("  " + "-" * 48)

    max_error = 0.0
    for info_set in sorted(known.keys()):
        learned = solver.get_average_strategy(info_set)
        target = np.array(known[info_set])
        error = np.abs(learned - target).max()
        max_error = max(max_error, error)

        # Only show the Bet probability (Pass = 1 - Bet)
        print(f"  {info_set:8s}  {'Bet':6s}  {learned[1]:8.4f}  {target[1]:8.4f}  {error:8.4f}")

    print(f"\n  Max absolute error: {max_error:.6f}")
    print(f"  (All equilibria in [0, 1/3] yield game value = -1/18 ≈ -0.0556 for P1)")
    return max_error


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("LESSON 2: Kuhn Poker & Counterfactual Regret Minimization (CFR)")
    print("=" * 65)

    # ── Demo 1: Run CFR ──
    print("\n\nDEMO 1: Training CFR on Kuhn Poker")
    print("-" * 65)
    print("Running vanilla CFR for 100,000 iterations...")
    print("Watch exploitability converge to 0 and strategies stabilize.")

    solver = KuhnCFR()
    solver.train(num_iterations=100_000)

    # ── Demo 2: Final strategy ──
    solver.print_final_strategy()

    # ── Demo 3: Compare with known Nash ──
    compare_with_known_equilibrium(solver)

    # ── Demo 4: Exploitability over iterations ──
    print("\n\n" + "=" * 65)
    print("DEMO 2: Exploitability Convergence Rate")
    print("=" * 65)
    print("Training fresh solver, measuring exploitability at each checkpoint...")

    fresh = KuhnCFR()
    checkpoints = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]
    prev = 0
    print(f"\n  {'Iterations':>12s}  {'Exploitability':>15s}  {'Rate':>12s}")
    print("  " + "-" * 45)
    for cp in checkpoints:
        fresh.train(cp - prev, verbose=False)
        prev = cp
        exploit = fresh.compute_exploitability()
        bound = 2.0 * np.sqrt(2 / cp)  # theoretical O(1/sqrt(T)) bound
        print(f"  {cp:>12,}  {exploit:>15.6f}  {f'≤ {bound:.4f}':>12s}")

    # ── Key Takeaways ──
    print("\n\n" + "=" * 65)
    print("KEY TAKEAWAYS")
    print("=" * 65)
    print("""
    1. CFR = regret matching at EVERY information set in the game tree.
       The algorithm from Lesson 1 is the atomic building block.

    2. INFORMATION SETS group game states that look identical to a player.
       Your strategy can only depend on what you KNOW (your card + history).

    3. COUNTERFACTUAL VALUES weight outcomes by the OPPONENT's reach
       probability, not your own. This "what if I tried to reach here?"
       is the key insight that makes CFR work in sequential games.

    4. The AVERAGE strategy converges to Nash Equilibrium.
       (Not the current strategy — same as Lesson 1.)

    5. Kuhn Poker's Nash Equilibrium reveals real poker concepts:
       - Player 1 BLUFFS with Jack (the worst hand) at a specific frequency
       - Player 1 VALUE BETS with King at a correlated frequency
       - Player 2 must CALL with Queen at the right frequency (pot odds)
       - Player 2 has a POSITIONAL ADVANTAGE (game value = -1/18 for P1)
       - The bluff:value-bet ratio (alpha : 3*alpha = 1:3) is not arbitrary;
         it makes the opponent indifferent to calling.

    WHAT'S NEXT (Lesson 3):
    Leduc Poker adds a community card and multiple betting rounds.
    We'll see CFR scale to larger game trees with multi-street decisions.
    """)
