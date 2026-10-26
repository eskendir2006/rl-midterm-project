"""
agents.py - the three agents.

Every agent has   act(state, rng) -> action
which is the policy used during EVALUATION (no exploration, no learning).
rng is a numpy Generator owned by the evaluation code; only the Random agent
and SARSA's tie-breaking use it.

SarsaAgent additionally has the training methods epsilon_greedy() and update().
"""

import numpy as np

from env import N_ACTIONS, BASKET, UP, DOWN, LEFT, RIGHT, TAKE, DROP


class RandomAgent:
    """Baseline: picks one of the 6 actions uniformly at random."""

    def act(self, state, rng):
        return int(rng.integers(N_ACTIONS))


class GreedyAgent:
    """Fixed-rule (non-RL) baseline.

    Not carrying: walk toward the apple (fix the x difference first, then y), take it.
    Carrying:     walk toward the basket the same way, drop it.
    """

    def act(self, state, rng):
        agent_x, agent_y, apple_x, apple_y, carrying, timer = state

        if carrying:
            target_x, target_y = BASKET
            final_action = DROP
        else:
            target_x, target_y = apple_x, apple_y
            final_action = TAKE

        if agent_x < target_x:
            return RIGHT
        if agent_x > target_x:
            return LEFT
        if agent_y < target_y:
            return UP
        if agent_y > target_y:
            return DOWN
        return final_action          # we are on the target cell


class SarsaAgent:
    """Tabular SARSA (on-policy TD control) with an epsilon-greedy behaviour policy.

    Q is a numpy array of shape (5, 5, 5, 5, 2, max_timer + 1, 6):
      Q[agent_x, agent_y, apple_x, apple_y, carrying, timer, action]
    so Q[state] gives the 6 action values of a state and Q[state + (action,)]
    gives one value.
    """

    def __init__(self, max_timer, alpha, gamma, seed):
        self.alpha = alpha
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)   # used for exploration during training
        self.Q = np.zeros((5, 5, 5, 5, 2, max_timer + 1, N_ACTIONS))

    # ------------------------------------------------------------------
    def greedy_action(self, state, rng):
        """argmax_a Q(state, a); ties are broken uniformly at random with rng."""
        q = self.Q[state]
        best = np.flatnonzero(q == q.max())      # all actions with the highest value
        if len(best) == 1:
            return int(best[0])
        return int(best[rng.integers(len(best))])

    def act(self, state, rng):
        """Evaluation policy: purely greedy (epsilon = 0)."""
        return self.greedy_action(state, rng)

    # ------------------------------------------------------------------
    def epsilon_greedy(self, state, epsilon):
        """Behaviour policy used during training.
        With probability epsilon pick a random action, otherwise the greedy one."""
        if self.rng.random() < epsilon:
            return int(self.rng.integers(N_ACTIONS))
        return self.greedy_action(state, self.rng)

    # ------------------------------------------------------------------
    def update(self, s, a, r, s_next, a_next, done):
        """The SARSA update:

            Q(S,A) <- Q(S,A) + alpha * [ R + gamma * Q(S',A') - Q(S,A) ]

        If S' is the end of the episode (time limit), the target is just R
        (no bootstrap). The time limit is not a true terminal state; treating it
        as one is a small approximation.
        """
        if done:
            target = r                                        # TD target at the episode end
        else:
            target = r + self.gamma * self.Q[s_next + (a_next,)]  # TD target R + gamma*Q(S',A')
        td_error = target - self.Q[s + (a,)]                  # how wrong the current estimate is
        self.Q[s + (a,)] += self.alpha * td_error
