"""
=============================================================================
LESSON 3: Leduc Poker & Multi-Street CFR
Game: Leduc Hold'em (2-player, 6-card deck)
=============================================================================

THEORY
------
In Lesson 2, we implemented CFR for Kuhn Poker — a single betting round with
3 cards. Leduc Poker (Southey et al., 2005) adds two critical features that
bridge toward real poker:

    1. A COMMUNITY CARD dealt between two betting rounds
    2. MULTIPLE BETTING ROUNDS (streets) with different bet sizes

These two features introduce key strategic concepts absent from Kuhn:
    - Hand strength changes across streets (a pair on round 2 is unbeatable)
    - Semibluffing (betting with a draw that might improve)
    - Position advantage compounds across streets
    - Pot control / pot commitment decisions

LEDUC POKER RULES:
    Deck:   6 cards — J♠ J♦ Q♠ Q♦ K♠ K♦ (two suits of three ranks)
    Setup:  Both players ante 1 chip each. Each is dealt 1 private card.

    ROUND 1 (pre-flop):
        - Player 1 acts first: Check or Bet (2 chips)
        - Standard poker betting: check, bet, call, fold, raise
        - Maximum of 2 bets per round (bet + raise, or check-bet-raise)

    COMMUNITY CARD:
        - One card is dealt face-up from the remaining deck

    ROUND 2 (post-flop):
        - Player 1 acts first again
        - Same betting structure, but bets are 4 chips (double the round 1 size)
        - Maximum of 2 bets per round

    SHOWDOWN:
        - If a player's card matches the community card: that's a PAIR (best hand)
        - If both or neither player pairs, higher card wins
        - Pair > No Pair. Among no-pair hands: K > Q > J.

GAME SIZE:
    Leduc has ~936 information sets (depending on exact betting rules),
    compared to Kuhn's 12. This is a ~78x increase — still tractable for
    vanilla CFR but large enough to see real scaling behavior.

    Full game tree has approximately 11,000+ terminal nodes.

INFORMATION SETS:
    A player's info set = their private card + community card (if dealt)
                         + the full betting history across both rounds.
    Example: "Qs|Kd:cr/b" means:
        - Private card: Queen of spades
        - Community card: King of diamonds
        - Round 1 history: check, raise
        - "/" separates rounds
        - Round 2 history: bet

    Since suits don't affect hand strength, we can use rank-only abstraction:
    "Q|K:cr/b" — this reduces information sets by collapsing identical ranks.

KEY STRATEGIC CONCEPTS:
    1. SEMIBLUFFING: Betting with a Queen when the flop hasn't come yet.
       If a Queen comes on the board, you make a pair. Unlike Kuhn's pure
       bluffs (Jack can never improve), Leduc has semi-bluffs with equity.

    2. HAND STRENGTH REVERSAL: You hold a King pre-flop (strong). A Queen
       hits the board and the opponent bets. Your King is now second-best
       if they have a Queen. This dynamic of "made hand vs. draw" is absent
       from Kuhn Poker.

    3. MULTI-STREET VALUE: Getting two betting rounds means you can extract
       more chips with strong hands. But it also means the pot grows, making
       bluffs more expensive and calls more committing.

    4. POSITION COMPOUNDS: Being in position (Player 2) for TWO streets of
       information is much more valuable than for one.

=============================================================================
"""

import numpy as np
from collections import defaultdict
import itertools


# ── Game Definition ────────────────────────────────────────────────────────

# Ranks (suits are irrelevant for hand strength in Leduc)
JACK, QUEEN, KING = 0, 1, 2
RANK_NAMES = {0: "J", 1: "Q", 2: "K"}

# Full deck: two cards of each rank
DECK = [0, 0, 1, 1, 2, 2]  # J, J, Q, Q, K, K

# Betting parameters
ANTE = 1
BET_SIZES = [2, 4]  # Round 1 bet = 2, Round 2 bet = 4
MAX_BETS_PER_ROUND = 2  # bet + raise allowed, no re-raise


class LeducState:
    """
    Represents a state in the Leduc Poker game tree.

    This class handles the game logic: determining whose turn it is,
    what actions are available, whether the hand is over, and what
    the terminal payoff is.
    """

    def __init__(self, cards, board=None, history=None, pot=None):
        """
        Parameters
        ----------
        cards : list of int
            cards[0] = Player 0's card, cards[1] = Player 1's card (rank values)
        board : int or None
            Community card rank, None if not yet dealt
        history : list of list
            history[0] = round 1 actions, history[1] = round 2 actions
            Each action is 'k' (check), 'b' (bet/raise), 'c' (call), 'f' (fold)
        pot : list of int
            pot[0] = Player 0's total investment, pot[1] = Player 1's investment
        """
        self.cards = cards
        self.board = board
        self.history = history if history is not None else [[]]
        self.pot = pot if pot is not None else [ANTE, ANTE]

    @property
    def current_round(self):
        """0-indexed round number."""
        return len(self.history) - 1

    @property
    def is_terminal(self):
        """Check if the hand is over."""
        # Fold
        cur = self.history[self.current_round]
        if len(cur) >= 1 and cur[-1] == 'f':
            return True
        # Both rounds complete
        if self.current_round == 1 and self._round_complete(1):
            return True
        return False

    @property
    def is_chance_node(self):
        """Check if we need to deal the community card."""
        return (self.current_round == 0 and self._round_complete(0)
                and self.board is None
                and not self.is_terminal)

    def _round_complete(self, round_idx):
        """Check if a betting round is complete."""
        actions = self.history[round_idx]
        if len(actions) < 2:
            return False
        # Check-check
        if actions[-2:] == ['k', 'k']:
            return True
        # Bet-call or bet-fold, raise-call or raise-fold
        if actions[-1] in ('c', 'f') and actions[-2] == 'b':
            return True
        return False

    @property
    def acting_player(self):
        """Which player acts next."""
        cur = self.history[self.current_round]
        return len(cur) % 2  # P0 acts first in each round

    def get_actions(self):
        """Return available actions for the current player."""
        cur = self.history[self.current_round]
        num_bets = sum(1 for a in cur if a == 'b')

        if len(cur) == 0:
            # First action in round: check or bet
            return ['k', 'b']

        last = cur[-1]
        if last == 'k':
            # Opponent checked: check or bet
            return ['k', 'b']
        elif last == 'b':
            if num_bets < MAX_BETS_PER_ROUND:
                # Facing a bet with raises remaining: fold, call, or raise
                return ['f', 'c', 'b']
            else:
                # Facing a bet, max bets reached: fold or call only
                return ['f', 'c']

        return ['k', 'b']

    def apply_action(self, action):
        """Return a new state after applying the given action."""
        new_history = [list(r) for r in self.history]
        new_pot = list(self.pot)
        player = self.acting_player
        round_idx = self.current_round
        bet_size = BET_SIZES[round_idx]

        if action == 'b':
            # Bet or raise
            num_bets = sum(1 for a in new_history[round_idx] if a == 'b')
            new_pot[player] = new_pot[1 - player] + bet_size
            new_history[round_idx].append('b')
        elif action == 'c':
            # Call: match opponent's investment
            new_pot[player] = new_pot[1 - player]
            new_history[round_idx].append('c')
        elif action == 'f':
            new_history[round_idx].append('f')
        elif action == 'k':
            new_history[round_idx].append('k')

        return LeducState(self.cards, self.board, new_history, new_pot)

    def deal_board(self, board_card):
        """Return a new state with the community card dealt and round 2 started."""
        new_history = [list(r) for r in self.history]
        new_history.append([])  # Start round 2
        return LeducState(self.cards, board_card, new_history, list(self.pot))

    def get_terminal_value(self):
        """Return payoff for Player 0. (Player 1's payoff is the negative.)"""
        cur = self.history[self.current_round]

        # Fold
        if cur[-1] == 'f':
            # Who folded? The player who just acted
            folder = (len(cur) - 1) % 2  # player index of last action
            # Folder loses their investment
            if folder == 0:
                return -self.pot[0]
            else:
                return self.pot[1]

        # Showdown
        winner = self._determine_winner()
        if winner == 0:
            return self.pot[1]  # P0 wins P1's investment
        elif winner == 1:
            return -self.pot[0]  # P0 loses their investment
        else:
            return 0  # Tie (shouldn't happen in standard Leduc)

    def _determine_winner(self):
        """0 if P0 wins, 1 if P1 wins, -1 if tie (split pot)."""
        p0_pair = (self.cards[0] == self.board)
        p1_pair = (self.cards[1] == self.board)

        if p0_pair and not p1_pair:
            return 0
        elif p1_pair and not p0_pair:
            return 1
        else:
            # Neither pairs (both pairing is impossible — only 2 cards per rank).
            # Higher card wins; equal cards split the pot.
            if self.cards[0] > self.cards[1]:
                return 0
            elif self.cards[1] > self.cards[0]:
                return 1
            else:
                return -1  # Tie — split pot

    def get_info_set(self):
        """
        Return the information set string for the acting player.

        Format: "RANK|BOARD:round1_actions/round2_actions"
        Board is "?" if not yet dealt.
        """
        player = self.acting_player
        card_str = RANK_NAMES[self.cards[player]]
        board_str = RANK_NAMES[self.board] if self.board is not None else ""

        round_strs = []
        for round_actions in self.history:
            round_strs.append("".join(round_actions))

        if board_str:
            return f"{card_str}|{board_str}:{'/'.join(round_strs)}"
        else:
            return f"{card_str}:{'/'.join(round_strs)}"


class LeducCFR:
    """
    Vanilla CFR for Leduc Poker.

    Same algorithm as Lesson 2's KuhnCFR, but applied to a larger game with:
    - Multiple betting rounds
    - A community card (chance node between rounds)
    - More actions per decision (check, bet, call, fold, raise)

    The CFR traversal is identical in structure. The only new element is
    handling the chance node (community card) by averaging over all possible
    board cards weighted by their probability.
    """

    def __init__(self):
        self.cumulative_regret = defaultdict(lambda: None)
        self.strategy_sum = defaultdict(lambda: None)
        self.num_iterations = 0
        self._info_set_actions = {}  # cache: info_set -> list of actions

    def _get_or_create(self, info_set, num_actions):
        """Lazily initialize regret/strategy arrays for an info set."""
        if self.cumulative_regret[info_set] is None:
            self.cumulative_regret[info_set] = np.zeros(num_actions)
            self.strategy_sum[info_set] = np.zeros(num_actions)

    def get_strategy(self, info_set):
        """Regret matching: convert cumulative regrets to a strategy."""
        regret = self.cumulative_regret[info_set]
        if regret is None:
            n = len(self._info_set_actions.get(info_set, [0, 1]))
            return np.ones(n) / n
        positive = np.maximum(regret, 0)
        total = positive.sum()
        if total > 0:
            return positive / total
        return np.ones(len(regret)) / len(regret)

    def get_average_strategy(self, info_set):
        """The average strategy — our Nash Equilibrium approximation."""
        s = self.strategy_sum[info_set]
        if s is None:
            n = len(self._info_set_actions.get(info_set, [0, 1]))
            return np.ones(n) / n
        total = s.sum()
        if total > 0:
            return s / total
        return np.ones(len(s)) / len(s)

    def cfr(self, state, reach_0, reach_1):
        """
        Recursive CFR traversal for Leduc Poker.

        The structure is identical to Kuhn CFR with two additions:
        1. Chance nodes: when the community card needs to be dealt,
           we average over all possible board cards.
        2. Variable action counts: some nodes have 2 actions (check/bet),
           others have 3 (fold/call/raise).
        """
        # ── Terminal node ──
        if state.is_terminal:
            return state.get_terminal_value()

        # ── Chance node: deal community card ──
        if state.is_chance_node:
            value = 0.0
            # Remaining cards in deck (exclude the two dealt to players)
            remaining = list(DECK)
            remaining.remove(state.cards[0])
            remaining.remove(state.cards[1])
            # Note: remaining may have duplicates (e.g., two Jacks left)
            # We need to iterate over unique cards with correct probabilities
            unique_remaining = list(set(remaining))
            for board_card in unique_remaining:
                count = remaining.count(board_card)
                prob = count / len(remaining)
                new_state = state.deal_board(board_card)
                # IMPORTANT: Multiply reaches by chance probability.
                # The counterfactual regret formula requires π_{-i}(h)
                # to include chance probabilities. Without this, round 2
                # regrets aren't weighted correctly across different
                # community cards.
                value += prob * self.cfr(new_state,
                                        reach_0 * prob,
                                        reach_1 * prob)
            return value

        # ── Decision node ──
        player = state.acting_player
        info_set = state.get_info_set()
        actions = state.get_actions()
        num_actions = len(actions)

        # Cache actions for this info set
        self._info_set_actions[info_set] = actions

        # Initialize arrays if needed
        self._get_or_create(info_set, num_actions)

        strategy = self.get_strategy(info_set)
        action_values = np.zeros(num_actions)
        node_value = 0.0

        for i, action in enumerate(actions):
            new_state = state.apply_action(action)
            if player == 0:
                action_values[i] = self.cfr(
                    new_state, reach_0 * strategy[i], reach_1)
            else:
                action_values[i] = self.cfr(
                    new_state, reach_0, reach_1 * strategy[i])
            node_value += strategy[i] * action_values[i]

        # ── Regret update ──
        opponent_reach = reach_1 if player == 0 else reach_0
        my_reach = reach_0 if player == 0 else reach_1

        for i in range(num_actions):
            regret = action_values[i] - node_value
            if player == 1:
                regret = -regret
            self.cumulative_regret[info_set][i] += opponent_reach * regret

        # ── Strategy accumulation ──
        self.strategy_sum[info_set] += my_reach * strategy

        return node_value

    def train(self, num_iterations, verbose=True):
        """
        Run CFR for the specified number of iterations.

        Each iteration enumerates all 30 possible physical card deals
        (6 cards × 5 remaining = 30 ordered pairs). Since cards of the
        same rank are strategically identical, some deals produce
        duplicate traversals — this is fine and correctly weights them.
        """
        # All 30 physical deals: pick any 2 of the 6 cards (ordered)
        all_deals = []
        for i in range(len(DECK)):
            for j in range(len(DECK)):
                if i != j:
                    all_deals.append((DECK[i], DECK[j]))

        for iteration in range(num_iterations):
            for c0, c1 in all_deals:
                state = LeducState([c0, c1])
                self.cfr(state, 1.0, 1.0)

            self.num_iterations += 1

            if verbose and (iteration + 1) in {1, 10, 100, 500, 1000}:
                self._print_progress(iteration + 1)

        return self

    def _print_progress(self, iteration):
        """Print exploitability at checkpoints."""
        exploit = self.compute_exploitability()
        n_info_sets = sum(1 for v in self.cumulative_regret.values()
                         if v is not None)
        print(f"  Iteration {iteration:>6,}  |  Exploitability: {exploit:.6f}"
              f"  |  Info sets: {n_info_sets}")

    def compute_exploitability(self):
        """
        Compute exploitability of the current average strategy.

        exploitability = (BR_value_P0 + BR_value_P1) / 2
        """
        all_deals = []
        for i in range(len(DECK)):
            for j in range(len(DECK)):
                if i != j:
                    all_deals.append((DECK[i], DECK[j]))

        br_val_0 = self._compute_br_value(all_deals, br_player=0)
        br_val_1 = self._compute_br_value(all_deals, br_player=1)
        return max(0.0, (br_val_0 + br_val_1) / 2.0)

    def _compute_br_value(self, deals, br_player):
        """
        Compute best-response value for br_player against the opponent's
        average strategy, respecting information set constraints.

        Two-pass approach:
        1. Traverse each deal collecting action EVs at every BR info set
        2. Pick best action per info set and evaluate the final expected value

        The BR player must make the SAME decision at all game states within
        the same info set — they can't see the opponent's card.
        """
        # Pass 1: Collect action EVs for all BR info sets across all deals
        info_set_evs = {}  # info_set -> np.array of action EVs

        for c0, c1 in deals:
            state = LeducState([c0, c1])
            self._collect_br_action_values(
                state, br_player, info_set_evs, 1.0)

        # Pick best action per info set
        br_strategy = {}
        for info_set, evs in info_set_evs.items():
            best = np.argmax(evs)
            s = np.zeros(len(evs))
            s[best] = 1.0
            br_strategy[info_set] = s

        # Pass 2: Evaluate the tree with BR strategy
        total = 0.0
        for c0, c1 in deals:
            state = LeducState([c0, c1])
            total += self._eval_tree(state, br_player, br_strategy)
        return total / len(deals)

    def _collect_br_action_values(self, state, br_player, evs_dict, opp_reach):
        """
        Single-pass traversal that collects action EVs at every BR info set.

        opp_reach: product of opponent's strategy probs and chance probs on
                   the path to this node (i.e., π_{-i}(h) excluding initial deal).

        At opponent nodes: multiply opp_reach by their action probability.
        At chance nodes: multiply opp_reach by chance probability.
        At BR player nodes: opp_reach unchanged (BR player's own probs
            don't affect the opponent reach). Record weighted action EVs.
            Return avg-strategy-weighted value upward.
        """
        if state.is_terminal:
            val = state.get_terminal_value()
            return val if br_player == 0 else -val

        if state.is_chance_node:
            remaining = list(DECK)
            remaining.remove(state.cards[0])
            remaining.remove(state.cards[1])
            unique_remaining = list(set(remaining))
            value = 0.0
            for board_card in unique_remaining:
                count = remaining.count(board_card)
                prob = count / len(remaining)
                new_state = state.deal_board(board_card)
                value += prob * self._collect_br_action_values(
                    new_state, br_player, evs_dict, opp_reach * prob)
            return value

        player = state.acting_player
        info_set = state.get_info_set()
        actions = state.get_actions()

        if player == br_player:
            # Evaluate each action and record weighted EVs
            action_vals = np.zeros(len(actions))
            for i, action in enumerate(actions):
                new_state = state.apply_action(action)
                action_vals[i] = self._collect_br_action_values(
                    new_state, br_player, evs_dict, opp_reach)

            # Accumulate into the info set's aggregated action EVs
            if info_set not in evs_dict:
                evs_dict[info_set] = np.zeros(len(actions))
            evs_dict[info_set] += opp_reach * action_vals

            # Return the avg-strategy-weighted value for parent computation
            avg = self.get_average_strategy(info_set)
            return float(avg @ action_vals)
        else:
            # Opponent plays average strategy — multiply opp_reach by prob
            strategy = self.get_average_strategy(info_set)
            value = 0.0
            for i, action in enumerate(actions):
                new_state = state.apply_action(action)
                value += strategy[i] * self._collect_br_action_values(
                    new_state, br_player, evs_dict,
                    opp_reach * strategy[i])
            return value

    def _eval_tree(self, state, br_player, br_strategy):
        """
        Evaluate the game tree where br_player follows br_strategy
        and the opponent follows their average strategy.
        Returns value from br_player's perspective.
        """
        if state.is_terminal:
            val = state.get_terminal_value()
            return val if br_player == 0 else -val

        if state.is_chance_node:
            remaining = list(DECK)
            remaining.remove(state.cards[0])
            remaining.remove(state.cards[1])
            unique_remaining = list(set(remaining))
            value = 0.0
            for board_card in unique_remaining:
                count = remaining.count(board_card)
                prob = count / len(remaining)
                new_state = state.deal_board(board_card)
                value += prob * self._eval_tree(new_state, br_player, br_strategy)
            return value

        player = state.acting_player
        info_set = state.get_info_set()
        actions = state.get_actions()

        if player == br_player:
            strategy = br_strategy.get(info_set)
            if strategy is None:
                strategy = np.ones(len(actions)) / len(actions)
        else:
            strategy = self.get_average_strategy(info_set)

        value = 0.0
        for i, action in enumerate(actions):
            new_state = state.apply_action(action)
            value += strategy[i] * self._eval_tree(new_state, br_player, br_strategy)
        return value

    def print_strategy(self, round_filter=None, card_filter=None):
        """Print strategies at all information sets, organized by round and card."""
        print("\n" + "=" * 70)
        print("LEARNED STRATEGY (Nash Equilibrium Approximation)")
        print("=" * 70)

        info_sets = sorted(
            [k for k in self._info_set_actions.keys()
             if self.strategy_sum[k] is not None],
            key=lambda k: (len(k), k)
        )

        # Group by player and round
        for player in [0, 1]:
            label = "Player 1 (first to act)" if player == 0 else "Player 2"
            print(f"\n  {label}:")

            for info_set in info_sets:
                # Determine player from the history
                actions_in_round = info_set.split(":")[-1] if ":" in info_set else ""
                current_actions = actions_in_round.split("/")[-1] if "/" in actions_in_round else actions_in_round
                is_player = len(current_actions) % 2 == player

                if not is_player:
                    continue

                avg = self.get_average_strategy(info_set)
                actions = self._info_set_actions[info_set]

                # Format strategy string
                parts = []
                for i, a in enumerate(actions):
                    action_name = {'k': 'Check', 'b': 'Bet', 'c': 'Call',
                                   'f': 'Fold'}[a]
                    pct = avg[i] * 100
                    parts.append(f"{action_name}={pct:5.1f}%")

                print(f"    {info_set:20s}  {' | '.join(parts)}")


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("LESSON 3: Leduc Poker & Multi-Street CFR")
    print("=" * 70)

    # ── Demo 1: Train CFR with exploitability tracking ──
    print("\n\nDEMO 1: Training CFR on Leduc Poker")
    print("-" * 70)
    print("Running vanilla CFR (288 info sets — 24x more than Kuhn)...\n")

    fresh = LeducCFR()
    fresh.train(num_iterations=1000)

    # ── Demo 2: Strategy Analysis ──
    print("\n\n" + "=" * 70)
    print("DEMO 2: Learned Strategy Analysis")
    print("=" * 70)

    def show_strategy(solver, info_set, label=""):
        """Pretty-print the average strategy at an info set."""
        if info_set not in solver._info_set_actions:
            return
        avg = solver.get_average_strategy(info_set)
        actions = solver._info_set_actions[info_set]
        parts = []
        for i, a in enumerate(actions):
            name = {'k': 'Chk', 'b': 'Bet', 'c': 'Call', 'f': 'Fold'}[a]
            pct = avg[i] * 100
            if pct > 0.5:
                parts.append(f"{name} {pct:5.1f}%")
        desc = f"  {info_set:20s}  {' | '.join(parts)}"
        if label:
            desc += f"    ← {label}"
        print(desc)

    print("\n  ── Round 1: Opening Actions (Player 1) ──")
    show_strategy(fresh, "J:", "Worst hand — mostly check")
    show_strategy(fresh, "Q:", "Middle hand")
    show_strategy(fresh, "K:", "Best hand — bet for value")

    print("\n  ── Round 1: Facing Check (Player 2) ──")
    show_strategy(fresh, "J:k", "Weakest — bluff sometimes?")
    show_strategy(fresh, "Q:k", "Middle")
    show_strategy(fresh, "K:k", "Strong — bet for value")

    print("\n  ── Round 1: Facing Bet (Player 2) ──")
    show_strategy(fresh, "J:b", "Worst hand facing aggression")
    show_strategy(fresh, "Q:b", "Call/raise with equity")
    show_strategy(fresh, "K:b", "Best hand — raise?")

    print("\n  ── Round 2: Opening with a PAIR (Player 1, after check-check) ──")
    show_strategy(fresh, "J|J:kk/", "Pair of Jacks (weakest pair)")
    show_strategy(fresh, "Q|Q:kk/", "Pair of Queens")
    show_strategy(fresh, "K|K:kk/", "Pair of Kings (the nuts)")

    print("\n  ── Round 2: Opening without a pair (Player 1, after check-check) ──")
    show_strategy(fresh, "J|Q:kk/", "J with Q on board")
    show_strategy(fresh, "J|K:kk/", "J with K on board (worst hand)")
    show_strategy(fresh, "Q|J:kk/", "Q with J on board")
    show_strategy(fresh, "Q|K:kk/", "Q with K on board")
    show_strategy(fresh, "K|J:kk/", "K with J on board")
    show_strategy(fresh, "K|Q:kk/", "K with Q on board (best unpaired)")

    # ── Demo 3: Compare game sizes ──
    print("\n\n" + "=" * 70)
    print("GAME SIZE COMPARISON")
    print("=" * 70)

    n_info_sets = sum(1 for v in fresh.cumulative_regret.values()
                      if v is not None)
    print(f"""
    Game              Info Sets     Terminal Nodes     Game Value (P1)
    ─────────────────────────────────────────────────────────────────
    Rock-Paper-Scissors    1          9                0.000
    Kuhn Poker            12         30               -0.056 (-1/18)
    Leduc Poker         {n_info_sets:>4}       ~1,400             ~-0.086
    NLHE (abstracted)   ~10^12     ~10^160            varies
    """)

    # ── Key Takeaways ──
    print("=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
    1. MULTI-STREET GAMES introduce chance nodes between betting rounds.
       CFR handles this by averaging over all possible community cards,
       weighted by their probability. The counterfactual regret formula
       requires these chance probabilities in the reach calculation.

    2. INFORMATION SETS grow dramatically with more streets and actions.
       Kuhn has 12 info sets, Leduc has 288. NLHE has ~10^14.
       But the CFR update at each info set is still the same simple
       regret matching from Lesson 1.

    3. HAND STRENGTH CHANGES across streets. In Kuhn, your hand never
       improves. In Leduc, a Queen can become a pair (the nuts!) when
       a Queen hits the board. This creates SEMIBLUFFING — betting
       with a hand that's currently behind but might improve.

    4. BET SIZING matters. Leduc uses 2-chip bets on round 1 and
       4-chip bets on round 2. The larger round-2 bets mean the pot
       grows faster, changing the risk/reward calculations.

    5. CONVERGENCE IS SLOWER for larger games, but the O(1/sqrt(T)) rate
       still applies. Each iteration touches more info sets, so the
       per-iteration cost is higher.

    WHAT'S NEXT (Lesson 4):
    Real poker has 52 cards, 4 betting rounds, and millions of possible
    bet sizes. We'll learn how ABSTRACTION (card bucketing, action
    abstraction) makes these massive games tractable for CFR.
    """)
