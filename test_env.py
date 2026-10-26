"""
test_env.py - plain-assert sanity tests of the environment rules.

    python test_env.py      -> prints "all tests passed"

Each test builds an environment, then sets the agent / apple / timer by hand so
the situation being tested is exactly known.
"""

import numpy as np

from env import (AppleGridEnv, BASKET, SLACK, T, UP, DOWN, LEFT, RIGHT, TAKE, DROP,
                 STEP_PENALTY, PICK_REWARD, DELIVER_REWARD, EXPIRE_PENALTY)


def make_env(agent, apple, timer=10, carrying=False):
    """An environment in a hand-made situation."""
    env = AppleGridEnv(seed=123)
    env.reset()
    env.agent_x, env.agent_y = agent
    env.apple_x, env.apple_y = apple
    env.carrying = carrying
    env.timer = 0 if carrying else timer
    return env


def close(a, b):
    return abs(a - b) < 1e-9


def check_new_apple(env):
    """A freshly spawned apple is not on the basket or the agent, and its timer is
    the Manhattan distance + SLACK."""
    apple = (env.apple_x, env.apple_y)
    assert apple != BASKET
    assert apple != (env.agent_x, env.agent_y)
    dist = abs(env.agent_x - env.apple_x) + abs(env.agent_y - env.apple_y)
    assert env.timer == dist + SLACK
    assert not env.carrying


def test_move_off_grid_keeps_position():
    env = make_env(agent=(0, 0), apple=(4, 4))
    for action in [LEFT, DOWN]:
        state, r, done, info = env.step(action)
        assert (env.agent_x, env.agent_y) == (0, 0)
        assert close(r, STEP_PENALTY)            # still costs the step penalty
    env = make_env(agent=(4, 4), apple=(0, 0))
    for action in [RIGHT, UP]:
        env.step(action)
        assert (env.agent_x, env.agent_y) == (4, 4)


def test_normal_move_and_timer_countdown():
    env = make_env(agent=(0, 0), apple=(4, 4), timer=10)
    state, r, done, info = env.step(UP)
    assert (env.agent_x, env.agent_y) == (0, 1)
    assert env.timer == 9                        # apple on the ground loses one unit
    assert state == (0, 1, 4, 4, 0, 9)


def test_take_only_on_apple_cell():
    env = make_env(agent=(0, 0), apple=(1, 0), timer=10)
    state, r, done, info = env.step(TAKE)        # not on the apple: nothing happens
    assert not env.carrying and not info["picked"]
    assert close(r, STEP_PENALTY)

    env.step(RIGHT)                              # now on the apple
    state, r, done, info = env.step(TAKE)
    assert env.carrying and info["picked"]
    assert close(r, STEP_PENALTY + PICK_REWARD)
    assert state == (1, 0, 1, 0, 1, 0)           # apple at agent, timer stored as 0

    state, r, done, info = env.step(TAKE)        # already carrying: no second pick reward
    assert not info["picked"]
    assert close(r, STEP_PENALTY)


def test_carried_apple_moves_and_timer_frozen():
    env = make_env(agent=(0, 0), apple=(0, 0), carrying=True)
    for _ in range(20):                          # far more steps than any timer
        state, r, done, info = env.step(UP if env.agent_y < 4 else DOWN)
        assert not info["expired"]
        assert state[2:4] == state[0:2] and state[4] == 1 and state[5] == 0


def test_drop_outside_basket_does_nothing():
    env = make_env(agent=(0, 0), apple=(0, 0), carrying=True)
    state, r, done, info = env.step(DROP)
    assert env.carrying                          # agent keeps the apple
    assert not info["delivered"]
    assert close(r, STEP_PENALTY)


def test_drop_without_apple_at_basket_does_nothing():
    env = make_env(agent=BASKET, apple=(4, 4), timer=10)
    state, r, done, info = env.step(DROP)
    assert not info["delivered"] and close(r, STEP_PENALTY)


def test_delivery_gives_reward_and_respawns():
    env = make_env(agent=BASKET, apple=BASKET, carrying=True)
    state, r, done, info = env.step(DROP)
    assert info["delivered"]
    assert close(r, STEP_PENALTY + DELIVER_REWARD)
    check_new_apple(env)                         # new apple, fresh timer (not decremented)


def test_timer_expiry_gives_penalty_and_respawns():
    env = make_env(agent=(0, 0), apple=(4, 4), timer=1)
    state, r, done, info = env.step(UP)
    assert info["expired"]
    assert close(r, STEP_PENALTY + EXPIRE_PENALTY)
    check_new_apple(env)


def test_reset_start_conditions():
    env = AppleGridEnv(seed=7)
    for _ in range(200):
        state = env.reset()
        assert (env.agent_x, env.agent_y) != BASKET
        assert env.t == 0
        check_new_apple(env)


def test_episode_ends_at_exactly_T():
    env = AppleGridEnv(seed=1)
    env.reset()
    for step in range(1, T + 1):
        state, r, done, info = env.step(UP)
        assert done == (step == T)


def test_same_seed_same_trajectory():
    def rollout(seed):
        env = AppleGridEnv(seed=seed)
        action_rng = np.random.default_rng(99)
        trajectory = [env.reset()]
        for _ in range(3 * T):
            state, r, done, info = env.step(int(action_rng.integers(6)))
            trajectory.append((state, r, done))
            if done:
                trajectory.append(env.reset())
        return trajectory
    assert rollout(5) == rollout(5)
    assert rollout(5) != rollout(6)


def test_include_timer_false_hides_timer():
    env = AppleGridEnv(seed=3, include_timer=False)
    state = env.reset()
    assert state[5] == 0 and env.timer > 0      # timer still runs inside the env
    for _ in range(T):
        state, r, done, info = env.step(UP)
        assert state[5] == 0


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print("ok  ", test.__name__)
    print("all tests passed")
