"""
replay.py - record evaluation episodes and build an interactive replay viewer.

    python replay.py        (about 30 s)  ->  results/replays.json
                                              results/replay_viewer.html  (open in a browser)

What it records:
  * SARSA (timer), seed 0, at several points during training ("generations"):
    the Q-table is copied at checkpoints 0, 500, 2000, ... 20000 episodes.
    Training is exactly the same as in run_all.py (same seeds, same schedule).
  * SARSA (no timer), seed 1, after training: the run whose greedy policy gets stuck.
  * The Random and Greedy baselines.

For every agent it replays the first N_EPISODES greedy evaluation episodes, using the
same evaluation seeds as experiment.evaluate(), so these are the actual episodes
behind the numbers in results/ (the script checks that the returns match).
"""

import json
import os

import numpy as np

from env import AppleGridEnv, ACTION_NAMES
from agents import RandomAgent, GreedyAgent, SarsaAgent
from experiment import (train_sarsa, evaluate, ALPHA, EVAL_SEED_OFFSET,
                        EVAL_POLICY_SEED_OFFSET)
from env import GAMMA

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

N_EPISODES = 6                                             # episodes recorded per agent
SNAPSHOTS = [0, 500, 2000, 5000, 8000, 10000, 15000, 20000]  # training episodes
N_TRAIN = 20000


def record_episodes(agent, seed, include_timer, has_q):
    """Replay the first N_EPISODES evaluation episodes step by step.

    Each step is stored as a list:
      [agent_x, agent_y, apple_x, apple_y, carrying, true_timer, action, reward, event]
    true_timer is the environment's real timer (even if the agent cannot see it),
    event is "" or a combination of "p" (picked), "d" (delivered), "e" (expired).
    If the agent has a Q-table, the 6 Q-values of the state it acted on are stored too.
    """
    env = AppleGridEnv(seed=seed + EVAL_SEED_OFFSET, include_timer=include_timer)
    policy_rng = np.random.default_rng(seed + EVAL_POLICY_SEED_OFFSET)

    episodes = []
    for _ in range(N_EPISODES):
        state = env.reset()
        start = [env.agent_x, env.agent_y, env.apple_x, env.apple_y, 0, env.timer]
        steps, qvalues = [], []
        done = False
        total = 0.0
        while not done:
            if has_q:
                qvalues.append([round(float(v), 3) for v in agent.Q[state]])
            action = agent.act(state, policy_rng)
            state, r, done, info = env.step(action)
            total += r
            event = ("p" if info["picked"] else "") + ("d" if info["delivered"] else "") \
                + ("e" if info["expired"] else "")
            steps.append([env.agent_x, env.agent_y, env.apple_x, env.apple_y,
                          int(env.carrying), env.timer, action, round(r, 2), event])
        episodes.append({"start": start, "steps": steps,
                         "q": qvalues if has_q else None, "return": round(total, 2)})
    return episodes


def check_matches_evaluate(agent, seed, include_timer, episodes):
    """The recorded returns must equal what experiment.evaluate() measures."""
    returns = evaluate(agent, N_EPISODES, seed, include_timer)[0]
    recorded = np.array([ep["return"] for ep in episodes])
    assert np.allclose(returns, recorded), (returns, recorded)


def sarsa_from_q(q, max_timer):
    """A SARSA agent holding a copy of a saved Q-table (only used to act greedily)."""
    agent = SarsaAgent(max_timer=max_timer, alpha=ALPHA, gamma=GAMMA, seed=0)
    agent.Q = q
    return agent


def main():
    max_timer = AppleGridEnv(seed=0).max_timer
    agents = []

    # SARSA (timer), seed 0: copy the Q-table at the snapshot checkpoints.
    saved = {}

    def save_q(episodes_trained, agent):
        if episodes_trained in SNAPSHOTS:
            saved[episodes_trained] = agent.Q.copy()

    print("training SARSA (timer), seed 0 ...")
    train_sarsa(0, N_TRAIN, include_timer=True, on_checkpoint=save_q)
    for n in SNAPSHOTS:
        agent = sarsa_from_q(saved[n], max_timer)
        episodes = record_episodes(agent, 0, True, has_q=True)
        check_matches_evaluate(agent, 0, True, episodes)
        agents.append({"id": f"sarsa_{n}", "group": "SARSA (timer), seed 0",
                       "label": f"{n:,} episodes", "episodes_trained": n,
                       "sees_timer": True, "episodes": episodes})

    # SARSA (no timer), seed 1: the configuration whose greedy policy gets stuck.
    print("training SARSA (no timer), seed 1 ...")
    agent, _, _ = train_sarsa(1, N_TRAIN, include_timer=False)
    episodes = record_episodes(agent, 1, False, has_q=True)
    check_matches_evaluate(agent, 1, False, episodes)
    agents.append({"id": "sarsa_no_timer_seed1", "group": "SARSA (no timer), seed 1",
                   "label": "20,000 episodes", "episodes_trained": N_TRAIN,
                   "sees_timer": False, "episodes": episodes})

    # Baselines, on the same evaluation seed as SARSA seed 0.
    for agent_id, label, agent in [("greedy", "Greedy (fixed rule)", GreedyAgent()),
                                   ("random", "Random", RandomAgent())]:
        episodes = record_episodes(agent, 0, True, has_q=False)
        check_matches_evaluate(agent, 0, True, episodes)
        agents.append({"id": agent_id, "group": "Baselines", "label": label,
                       "episodes_trained": None, "sees_timer": True, "episodes": episodes})

    data = {"actions": ACTION_NAMES, "agents": agents, "n_episodes": N_EPISODES}
    os.makedirs(RESULTS, exist_ok=True)
    text = json.dumps(data, separators=(",", ":"))
    with open(os.path.join(RESULTS, "replays.json"), "w", encoding="utf-8") as f:
        f.write(text)

    # Build a stand-alone viewer: the template with the data pasted in.
    with open(os.path.join(HERE, "viewer_template.html"), encoding="utf-8") as f:
        html = f.read()
    html = html.replace("/*__REPLAY_DATA__*/null", text)
    # The template is the page body only; add the document header for a local file.
    html = ('<!doctype html>\n<html lang="en">\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            + html + "\n</html>\n")
    with open(os.path.join(RESULTS, "replay_viewer.html"), "w", encoding="utf-8") as f:
        f.write(html)

    for a in agents:
        returns = [ep["return"] for ep in a["episodes"]]
        print(f"{a['group']:<26} {a['label']:<20} returns of recorded episodes: {returns}")
    print("wrote results/replays.json and results/replay_viewer.html")


if __name__ == "__main__":
    main()
