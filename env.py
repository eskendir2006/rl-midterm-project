"""
env.py - the Apple Collector gridworld.

A single agent moves on a 5x5 grid. There is always exactly one apple, either
lying on the ground or being carried by the agent. The agent must pick the apple
up ("take") and carry it to the basket in the centre ("drop") before the apple's
timer runs out. A new apple appears as soon as the old one is delivered or expires.

All randomness (where the agent starts, where apples appear) comes from one
numpy.random.Generator created from the seed given to the constructor.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Constants (change them here)
# ---------------------------------------------------------------------------
N = 5                    # grid is N x N, cells (x, y) with x, y in {0, ..., N-1}
BASKET = (2, 2)          # fixed basket position (centre of the grid)
SLACK = 3                # extra steps added to the apple timer on spawn
T = 100                  # episode length (time limit) in steps
GAMMA = 0.95             # discount factor (used by the agent, listed here for reference)

STEP_PENALTY = -0.05     # every step, whatever the action
PICK_REWARD = 1.0        # successful take (only possible once per apple)
DELIVER_REWARD = 10.0    # successful drop at the basket
EXPIRE_PENALTY = -5.0    # apple timer reached 0 while the apple was on the ground

# Actions
UP, DOWN, LEFT, RIGHT, TAKE, DROP = 0, 1, 2, 3, 4, 5
N_ACTIONS = 6
ACTION_NAMES = ["up", "down", "left", "right", "take", "drop"]


class AppleGridEnv:
    """The Apple Collector environment.

    state = (agent_x, agent_y, apple_x, apple_y, carrying, timer)
      carrying is 0 or 1.
      While carrying, apple_x/apple_y equal the agent's position and timer = 0.
      If include_timer is False, the timer component is always reported as 0.
    """

    def __init__(self, seed, slack=SLACK, horizon=T, include_timer=True):
        self.rng = np.random.default_rng(seed)  # the only source of randomness
        self.slack = slack
        self.horizon = horizon
        self.include_timer = include_timer
        self.max_timer = (N - 1) * 2 + slack     # largest Manhattan distance (8) + slack

        # These are set properly in reset(); listed here so every attribute is visible.
        self.agent_x = 0
        self.agent_y = 0
        self.apple_x = 0
        self.apple_y = 0
        self.carrying = False
        self.timer = 0
        self.t = 0

    # ------------------------------------------------------------------
    def reset(self):
        """Start a new episode and return the first state."""
        # Agent starts on a uniformly random cell that is not the basket.
        free_cells = [(x, y) for x in range(N) for y in range(N) if (x, y) != BASKET]
        i = self.rng.integers(len(free_cells))
        self.agent_x, self.agent_y = free_cells[i]

        self.carrying = False
        self.t = 0
        self._spawn_apple()
        return self.get_state()

    # ------------------------------------------------------------------
    def _spawn_apple(self):
        """Place a new apple uniformly at random on a cell that is neither the
        basket nor the agent's current cell, and start its timer."""
        cells = [(x, y) for x in range(N) for y in range(N)
                 if (x, y) != BASKET and (x, y) != (self.agent_x, self.agent_y)]
        i = self.rng.integers(len(cells))
        self.apple_x, self.apple_y = cells[i]

        # Timer = Manhattan distance from agent to apple + slack.
        distance = abs(self.agent_x - self.apple_x) + abs(self.agent_y - self.apple_y)
        self.timer = distance + self.slack

    # ------------------------------------------------------------------
    def get_state(self):
        """Return the state as a tuple of ints, usable directly as a Q-table index."""
        if self.carrying:
            # The apple is in the agent's hands: same position, timer frozen (stored as 0).
            return (self.agent_x, self.agent_y, self.agent_x, self.agent_y, 1, 0)
        timer = self.timer if self.include_timer else 0
        return (self.agent_x, self.agent_y, self.apple_x, self.apple_y, 0, timer)

    # ------------------------------------------------------------------
    def step(self, action):
        """Apply one action. Returns (next_state, reward, done, info)."""
        picked = False
        delivered = False
        expired = False

        # 1. every step costs a small penalty
        r = STEP_PENALTY

        # 2. apply the action
        if action == UP:
            self.agent_y = min(self.agent_y + 1, N - 1)   # off-grid moves leave the agent in place
        elif action == DOWN:
            self.agent_y = max(self.agent_y - 1, 0)
        elif action == LEFT:
            self.agent_x = max(self.agent_x - 1, 0)
        elif action == RIGHT:
            self.agent_x = min(self.agent_x + 1, N - 1)
        elif action == TAKE:
            # Only works if not already carrying and standing on the apple.
            if not self.carrying and (self.agent_x, self.agent_y) == (self.apple_x, self.apple_y):
                self.carrying = True
                picked = True
                r += PICK_REWARD
        elif action == DROP:
            # Only works if carrying and standing on the basket.
            # Dropping anywhere else does nothing (the agent keeps the apple).
            if self.carrying and (self.agent_x, self.agent_y) == BASKET:
                self.carrying = False
                delivered = True
                r += DELIVER_REWARD

        # A carried apple moves with the agent.
        if self.carrying:
            self.apple_x, self.apple_y = self.agent_x, self.agent_y
            self.timer = 0

        # 3. after a delivery a new apple spawns immediately
        if delivered:
            self._spawn_apple()
        # 4. otherwise, an apple on the ground loses one unit of time
        elif not self.carrying:
            self.timer -= 1
            if self.timer == 0:
                r += EXPIRE_PENALTY
                expired = True
                self._spawn_apple()

        # 5. advance time; the episode ends exactly at the time limit
        self.t += 1
        done = self.t >= self.horizon

        # 6. return
        info = {"delivered": delivered, "expired": expired, "picked": picked}
        return self.get_state(), r, done, info
