# Apple Collector Gridworld: tabular SARSA

CSCI3235 Reinforcement Learning, Midterm I take-home project (Yergali Ussibaliyev).

An agent moves on a 5x5 grid, picks up one apple at a time and carries it to a
basket in the centre (2, 2) before the apple's timer runs out. The environment is
self-built (no Gymnasium). The learning method is tabular **SARSA** with an
ε-greedy behaviour policy. It is compared with a **Random** baseline and a
hand-coded **Greedy** rule (walk to the apple, take it, walk to the basket, drop it).

## Setup

- Python 3.10 or newer (developed and measured with Python 3.12.10 on Windows 11)
- Only numpy and matplotlib:

```
pip install -r requirements.txt
```

## How to run

Run from inside the `apple_rl/` folder:

```
python test_env.py          # environment rule tests, prints "all tests passed"
python run_all.py           # full run: all experiments + figures + summary
python run_all.py --quick   # smoke test: 2 seeds x 1000 episodes
python plots.py             # (optional) redraw the figures from the CSVs in results/
python replay.py            # (optional) interactive replay viewer -> results/replay_viewer.html
```

Measured runtime on the development machine (single process, 16-core desktop CPU):

| Command | Wall-clock time |
|---|---|
| `python run_all.py` | about 3 min 50 s (two measured runs: 3 min 46 s and 3 min 50 s) |
| `python run_all.py --quick` | about 8 s |
| `python test_env.py` | under 1 s |
| `python replay.py` | about 32 s |

`results/run_info.txt` records the exact time of the latest run, broken down by
configuration and seed.

## What the full run does

| ID | Experiment | Details |
|---|---|---|
| E1 | Training curve | SARSA (timer in state): every 500 training episodes, 100 greedy evaluation episodes |
| E2 | Final comparison | 500 greedy evaluation episodes per seed for SARSA, Random and Greedy |
| E3 | Ablation | SARSA with the timer removed from the state (`include_timer=False`) |
| E4 | Second ablation (optional) | SARSA with fixed ε = 0.1 instead of the decaying schedule |

SARSA settings: α = 0.1, γ = 0.95, ε decays linearly from 1.0 to 0.05 over the
first 80 % of episodes and then stays at 0.05, 20 000 training episodes per seed,
Q initialised to 0.

## Seeds

The experiment seeds are **0, 1, 2, 3, 4** (`--quick` uses 0 and 1). For an experiment seed `s`:

| Random stream | Seed |
|---|---|
| training environment | `s` |
| SARSA agent (exploration and tie-breaking during training) | `s + 5000` |
| evaluation environment (separate instance) | `s + 10000` |
| evaluation policy rng (Random agent's actions, SARSA tie-breaking) | `s + 20000` |

Every random number comes from a `numpy.random.Generator` created from these
seeds. The global `np.random` state and Python's `random` module are never used.
Running `run_all.py` twice gives an identical `summary.csv` and `per_seed.csv`.

## File guide

| File | What it contains |
|---|---|
| `env.py` | `AppleGridEnv`: `reset()`, `step(action)`. All constants (grid, rewards, SLACK, T) are at the top |
| `agents.py` | `RandomAgent`, `GreedyAgent`, `SarsaAgent` (ε-greedy, SARSA update) |
| `experiment.py` | `train_sarsa()`, `evaluate()`, the loop over seeds and configurations, CSV writing |
| `plots.py` | Builds every figure from the CSV files only |
| `run_all.py` | Single entry point; writes `results/run_info.txt` |
| `test_env.py` | Plain-assert tests of the environment rules |
| `replay.py` | Records greedy evaluation episodes of SARSA at training snapshots (0 to 20 000 episodes), the stuck no-timer seed-1 agent, Greedy and Random; builds the replay viewer |
| `viewer_template.html` | Page template for the replay viewer (`replay.py` pastes the recorded data into it) |
| `requirements.txt` | Pinned numpy and matplotlib versions |
| `report/report.md` | Report draft (figures linked from `../results/`) |

## What appears in `results/` after a run

The folder is deleted and rewritten on every run.

| File | Content |
|---|---|
| `summary.csv`, `summary.md` | Per agent: mean ± std (over seeds) of return, apples delivered, apples expired, fraction of stuck evaluation episodes, number of seeds, evaluation episodes per seed |
| `per_seed.csv` | One row per (agent, seed) with the seed-level means |
| `eval_curve.csv` | E1/E3/E4: mean greedy evaluation return every 500 training episodes, per configuration and seed |
| `curve_summary.csv` | Mean / std / min over seeds of the evaluation return at every checkpoint (used in the report text) |
| `training_windows.csv` | Mean training return in 10 equal windows of episodes (plateau check) |
| `training_returns.csv` | Raw return of every training episode (ε-greedy behaviour policy), one column per configuration and seed |
| `transition_example.csv` | One real `env.step` (state, action, reward, next state) used for fig0 |
| `fig0_transition_example.png` | Grid drawing of that transition |
| `fig1_training_curve.png` | (a) greedy evaluation return vs training episodes, SARSA timer / no timer, mean ± 1 std over 5 seeds, Random and Greedy as dashed lines; (b) smoothed training return |
| `fig2_final_comparison.png` | Bar charts of final evaluation return and apples delivered (mean ± std over seeds) |
| `fig3_epsilon_ablation.png` | E4: decaying vs fixed ε evaluation curves |
| `run_info.txt` | Python and package versions, seeds, hyperparameters, wall-clock runtime |
| `replays.json`, `replay_viewer.html` | Only after `python replay.py`: recorded episodes and a stand-alone viewer. Open the HTML file in any browser to watch the agent step by step, compare training snapshots, and see the Q-values behind every decision |
