"""
run_all.py - single entry point.

    python run_all.py            full run: 5 seeds x 20 000 episodes, all experiments + figures
    python run_all.py --quick    smoke test: 2 seeds x 1 000 episodes

Everything in results/ is overwritten on each run.
"""

import argparse
import os
import platform
import shutil
import sys
import time

import numpy as np
import matplotlib

import env
import experiment
import plots

FULL_SEEDS = [0, 1, 2, 3, 4]
FULL_EPISODES = 20000
QUICK_SEEDS = [0, 1]
QUICK_EPISODES = 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="2 seeds, 1000 episodes")
    args = parser.parse_args()

    seeds = QUICK_SEEDS if args.quick else FULL_SEEDS
    n_episodes = QUICK_EPISODES if args.quick else FULL_EPISODES
    configs = list(experiment.SARSA_CONFIGS)       # sarsa_timer, sarsa_no_timer, sarsa_fixed_eps

    out_dir = plots.RESULTS
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)                      # start from an empty results/ folder
    os.makedirs(out_dir)

    print(f"seeds={seeds}  episodes per seed={n_episodes}  configs={configs}")
    start = time.perf_counter()
    timings = experiment.run_experiments(seeds, n_episodes, out_dir, configs)
    experiment_seconds = time.perf_counter() - start

    plots.make_all_figures()
    total_seconds = time.perf_counter() - start

    # run_info.txt: everything needed to reproduce and check the run
    with open(os.path.join(out_dir, "run_info.txt"), "w", encoding="utf-8") as f:
        f.write(f"mode: {'quick' if args.quick else 'full'}\n")
        f.write(f"python: {sys.version.split()[0]}\n")
        f.write(f"numpy: {np.__version__}\n")
        f.write(f"matplotlib: {matplotlib.__version__}\n")
        f.write(f"platform: {platform.platform()}\n")
        f.write(f"processor: {platform.processor()}\n\n")
        f.write(f"seeds: {seeds}\n")
        f.write(f"training env seed = s, agent seed = s + {experiment.AGENT_SEED_OFFSET}, "
                f"evaluation env seed = s + {experiment.EVAL_SEED_OFFSET}, "
                f"evaluation policy rng seed = s + {experiment.EVAL_POLICY_SEED_OFFSET}\n")
        f.write(f"training episodes per seed: {n_episodes}\n")
        f.write(f"configs: {configs}\n\n")
        f.write(f"grid N={env.N}  basket={env.BASKET}  SLACK={env.SLACK}  T={env.T}\n")
        f.write(f"rewards: step={env.STEP_PENALTY} pick={env.PICK_REWARD} "
                f"deliver={env.DELIVER_REWARD} expire={env.EXPIRE_PENALTY}\n")
        f.write(f"alpha={experiment.ALPHA}  gamma={env.GAMMA}\n")
        f.write(f"epsilon: linear {experiment.EPS_START} -> {experiment.EPS_END} over first "
                f"{experiment.EPS_DECAY_FRACTION:.0%} of episodes, then constant "
                f"(sarsa_fixed_eps: constant 0.1)\n")
        f.write(f"evaluation: every {experiment.EVAL_EVERY} episodes x "
                f"{experiment.EVAL_EPISODES_CURVE} episodes; final {experiment.EVAL_EPISODES_FINAL} "
                f"episodes per seed; greedy policy (epsilon=0), no updates\n\n")
        f.write("wall-clock seconds per (config, seed) = training + all evaluations:\n")
        for (name, seed), sec in timings.items():
            f.write(f"  {name} seed {seed}: {sec:.1f}\n")
        f.write(f"experiments total: {experiment_seconds:.1f} s\n")
        f.write(f"total including figures: {total_seconds:.1f} s\n")

    print()
    print(f"{'agent':<18}{'mean return':>12}{'std':>8}{'delivered':>11}{'expired':>9}")
    for row in plots.read_csv("summary.csv"):
        print(f"{row['agent']:<18}{float(row['mean_return']):>12.2f}{float(row['std_return']):>8.2f}"
              f"{float(row['mean_delivered']):>11.2f}{float(row['mean_expired']):>9.2f}")
    print(f"done in {total_seconds:.1f} s; results in {out_dir}")


if __name__ == "__main__":
    main()
