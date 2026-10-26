"""
experiment.py - training, evaluation, and running everything over seeds.

How seeds are used (for one experiment seed s):
    training environment     AppleGridEnv(seed = s)
    SARSA agent exploration  SarsaAgent(seed = s + 5000)
    evaluation environment   AppleGridEnv(seed = s + 10000)   (a separate instance)
    evaluation tie-breaking  default_rng(s + 20000)           (also the Random agent's choices)
Every evaluation (the checkpoints during training and the final one) creates a
fresh evaluation environment from s + 10000, so all agents for seed s are tested
on the same sequence of start positions and apple spawns (as long as their
actions do not change what the environment draws).
"""

import csv
import os
import time

import numpy as np

from env import AppleGridEnv, SLACK, GAMMA
from agents import RandomAgent, GreedyAgent, SarsaAgent

# ---------------------------------------------------------------------------
# Default hyperparameters
# ---------------------------------------------------------------------------
ALPHA = 0.1
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_FRACTION = 0.8      # epsilon decays linearly over the first 80% of episodes

EVAL_EVERY = 500              # E1: evaluate the greedy policy every 500 training episodes
EVAL_EPISODES_CURVE = 100     # ... using 100 evaluation episodes
EVAL_EPISODES_FINAL = 500     # E2: final evaluation episodes per seed
STUCK_STEPS = 20              # an evaluation episode is "stuck" if the state is unchanged this long

AGENT_SEED_OFFSET = 5000
EVAL_SEED_OFFSET = 10000
EVAL_POLICY_SEED_OFFSET = 20000

# SARSA configurations. "sarsa_timer" is the main one; the others change ONE setting.
SARSA_CONFIGS = {
    "sarsa_timer":     {"include_timer": True,  "fixed_epsilon": None},   # main (E1, E2)
    "sarsa_no_timer":  {"include_timer": False, "fixed_epsilon": None},   # ablation E3
    "sarsa_fixed_eps": {"include_timer": True,  "fixed_epsilon": 0.1},    # optional ablation E4
}

# Names used in tables and figures.
DISPLAY_NAMES = {
    "random": "Random",
    "greedy": "Greedy (fixed rule)",
    "sarsa_timer": "SARSA (timer)",
    "sarsa_no_timer": "SARSA (no timer)",
    "sarsa_fixed_eps": "SARSA (timer, fixed eps=0.1)",
}


def epsilon_at(episode, n_episodes, fixed_epsilon=None):
    """Exploration rate for a training episode (episode counts from 0)."""
    if fixed_epsilon is not None:
        return fixed_epsilon
    decay_episodes = EPS_DECAY_FRACTION * n_episodes
    fraction = min(episode / decay_episodes, 1.0)
    return EPS_START + fraction * (EPS_END - EPS_START)


# ---------------------------------------------------------------------------
# Evaluation: separate environment, greedy policy, no learning
# ---------------------------------------------------------------------------
def evaluate(agent, n_episodes, seed, include_timer=True):
    """Run n_episodes with agent.act (no exploration, no Q updates).

    Returns four numpy arrays (one value per episode): undiscounted return,
    apples delivered, apples expired, and stuck (1 if the state stayed exactly the
    same for STUCK_STEPS consecutive steps, e.g. the greedy action is a "drop"
    away from the basket, which changes nothing, so the agent repeats it forever).
    """
    env = AppleGridEnv(seed=seed + EVAL_SEED_OFFSET, include_timer=include_timer)
    policy_rng = np.random.default_rng(seed + EVAL_POLICY_SEED_OFFSET)

    returns = np.zeros(n_episodes)
    delivered = np.zeros(n_episodes)
    expired = np.zeros(n_episodes)
    stuck = np.zeros(n_episodes)
    for ep in range(n_episodes):
        state = env.reset()
        done = False
        unchanged = 0                   # how many steps in a row the state did not change
        while not done:
            action = agent.act(state, policy_rng)
            next_state, r, done, info = env.step(action)
            returns[ep] += r
            delivered[ep] += info["delivered"]
            expired[ep] += info["expired"]
            unchanged = unchanged + 1 if next_state == state else 0
            if unchanged >= STUCK_STEPS:
                stuck[ep] = 1
            state = next_state
    return returns, delivered, expired, stuck


# ---------------------------------------------------------------------------
# Training: SARSA
# ---------------------------------------------------------------------------
def train_sarsa(seed, n_episodes, include_timer=True, fixed_epsilon=None, on_checkpoint=None):
    """Train one SARSA agent.

    on_checkpoint: optional function called as on_checkpoint(episodes_trained, agent) at
    every evaluation checkpoint (used by replay.py to copy the Q-table). It must not
    change the agent.

    Returns (agent, train_returns, curve):
      train_returns[i] = return of training episode i (behaviour policy, with exploration)
      curve = list of (episodes_trained, mean greedy evaluation return), starting at 0
    """
    env = AppleGridEnv(seed=seed, include_timer=include_timer)
    agent = SarsaAgent(max_timer=env.max_timer, alpha=ALPHA, gamma=GAMMA,
                       seed=seed + AGENT_SEED_OFFSET)

    train_returns = np.zeros(n_episodes)
    curve = []
    # Evaluation before any training (Q = 0, so ties everywhere -> random actions).
    curve.append((0, evaluate(agent, EVAL_EPISODES_CURVE, seed, include_timer)[0].mean()))
    if on_checkpoint is not None:
        on_checkpoint(0, agent)

    for ep in range(n_episodes):
        epsilon = epsilon_at(ep, n_episodes, fixed_epsilon)

        s = env.reset()
        a = agent.epsilon_greedy(s, epsilon)
        done = False
        while not done:
            s_next, r, done, info = env.step(a)
            train_returns[ep] += r
            if done:
                # Time limit reached: target is just R (no bootstrap).
                agent.update(s, a, r, None, None, True)
            else:
                # SARSA: choose A' from S' with the SAME epsilon-greedy policy, then update.
                a_next = agent.epsilon_greedy(s_next, epsilon)
                agent.update(s, a, r, s_next, a_next, False)
                s, a = s_next, a_next

        if (ep + 1) % EVAL_EVERY == 0:
            mean_eval = evaluate(agent, EVAL_EPISODES_CURVE, seed, include_timer)[0].mean()
            curve.append((ep + 1, mean_eval))
            if on_checkpoint is not None:
                on_checkpoint(ep + 1, agent)

    return agent, train_returns, curve


# ---------------------------------------------------------------------------
# Transition example for the report (Part B)
# ---------------------------------------------------------------------------
def find_transition_example(seed=0):
    """Roll out the Greedy agent and return the first step in which an apple is
    picked up: (state, action, reward, next_state). Taken from a real env.step."""
    env = AppleGridEnv(seed=seed)
    agent = GreedyAgent()
    rng = np.random.default_rng(seed)
    state = env.reset()
    done = False
    while not done:
        action = agent.act(state, rng)
        next_state, r, done, info = env.step(action)
        if info["picked"]:
            return state, action, r, next_state
        state = next_state
    return None


# ---------------------------------------------------------------------------
# Writing results
# ---------------------------------------------------------------------------
def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def run_experiments(seeds, n_episodes, out_dir, configs, log=print):
    """Run E1-E4 for all seeds and write the CSV files. Returns timing info."""
    os.makedirs(out_dir, exist_ok=True)

    per_seed_rows = []      # (agent, seed, mean_return, mean_delivered, mean_expired, eval_episodes, frac_stuck)
    curve_rows = []         # (agent, seed, episodes_trained, mean_eval_return)
    train_columns = {}      # column name -> per-episode training returns
    timings = {}

    def add_per_seed(name, seed, result):
        returns, delivered, expired, stuck = result
        per_seed_rows.append((name, seed, round(returns.mean(), 4), round(delivered.mean(), 4),
                              round(expired.mean(), 4), len(returns), round(stuck.mean(), 4)))

    # Baselines (E2): same evaluation seeds as SARSA.
    for seed in seeds:
        add_per_seed("random", seed, evaluate(RandomAgent(), EVAL_EPISODES_FINAL, seed))
        add_per_seed("greedy", seed, evaluate(GreedyAgent(), EVAL_EPISODES_FINAL, seed))
    log("baselines evaluated")

    # SARSA configurations (E1, E2, E3, E4).
    for name in configs:
        cfg = SARSA_CONFIGS[name]
        for seed in seeds:
            start = time.perf_counter()
            agent, train_returns, curve = train_sarsa(seed, n_episodes, cfg["include_timer"],
                                                      cfg["fixed_epsilon"])
            result = evaluate(agent, EVAL_EPISODES_FINAL, seed, cfg["include_timer"])
            seconds = time.perf_counter() - start
            timings[(name, seed)] = seconds

            add_per_seed(name, seed, result)
            for episodes_trained, mean_eval in curve:
                curve_rows.append((name, seed, episodes_trained, round(mean_eval, 4)))
            train_columns[f"{name}_seed{seed}"] = train_returns
            log(f"{name} seed {seed}: final eval return {result[0].mean():.2f}  ({seconds:.1f} s)")

    # per_seed.csv
    write_csv(os.path.join(out_dir, "per_seed.csv"),
              ["agent", "seed", "mean_return", "mean_delivered", "mean_expired", "eval_episodes",
               "frac_stuck"],
              per_seed_rows)

    # eval_curve.csv (E1)
    write_csv(os.path.join(out_dir, "eval_curve.csv"),
              ["agent", "seed", "episodes_trained", "mean_eval_return"], curve_rows)

    # training_returns.csv: one row per training episode, one column per (config, seed)
    names = list(train_columns)
    rows = []
    for ep in range(n_episodes):
        rows.append([ep + 1] + [f"{train_columns[c][ep]:.2f}" for c in names])
    write_csv(os.path.join(out_dir, "training_returns.csv"), ["episode"] + names, rows)

    # summary.csv and summary.md: mean and std over the seed-level means
    order = ["random", "greedy"] + list(configs)
    summary_rows = []
    for name in order:
        rows_for_agent = [r for r in per_seed_rows if r[0] == name]
        ret = np.array([r[2] for r in rows_for_agent])
        dlv = np.array([r[3] for r in rows_for_agent])
        exp = np.array([r[4] for r in rows_for_agent])
        stk = np.array([r[6] for r in rows_for_agent])
        summary_rows.append((name, DISPLAY_NAMES[name],
                             round(ret.mean(), 3), round(ret.std(ddof=1), 3),
                             round(dlv.mean(), 3), round(dlv.std(ddof=1), 3),
                             round(exp.mean(), 3), round(exp.std(ddof=1), 3),
                             len(rows_for_agent), EVAL_EPISODES_FINAL,
                             round(stk.mean(), 4), round(stk.std(ddof=1), 4)))
    header = ["agent", "display_name", "mean_return", "std_return", "mean_delivered",
              "std_delivered", "mean_expired", "std_expired", "n_seeds", "eval_episodes_per_seed",
              "frac_stuck", "std_frac_stuck"]
    write_csv(os.path.join(out_dir, "summary.csv"), header, summary_rows)

    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("| Agent | Mean return | Std (over seeds) | Apples delivered / ep | "
                "Apples expired / ep | Stuck episodes (%) | Seeds | Eval episodes / seed |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for r in summary_rows:
            f.write(f"| {r[1]} | {r[2]:.2f} | {r[3]:.2f} | {r[4]:.2f} ± {r[5]:.2f} | "
                    f"{r[6]:.2f} ± {r[7]:.2f} | {100 * r[10]:.1f} | {r[8]} | {r[9]} |\n")

    # transition_example.csv (for fig0 and the report)
    s, a, r, s2 = find_transition_example(seed=0)
    write_csv(os.path.join(out_dir, "transition_example.csv"),
              ["state", "action", "reward", "next_state"],
              [[" ".join(map(str, s)), a, round(r, 4), " ".join(map(str, s2))]])

    return timings
