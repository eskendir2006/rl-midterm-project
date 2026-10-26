"""
plots.py - builds every figure from the CSV files in results/ (never from memory).

    fig0_transition_example.png   one real env.step drawn on the grid
    fig1_training_curve.png       E1 + E3: greedy evaluation return during training
    fig2_final_comparison.png     E2 + E3: final evaluation return and apples delivered
    fig3_epsilon_ablation.png     E4 (optional): decaying vs fixed epsilon

Run on its own with:  python plots.py   (after run_all.py has written the CSVs)
"""

import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")            # draw to files, no window needed
import matplotlib.pyplot as plt

from env import BASKET, N, ACTION_NAMES

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# One fixed colour per agent, used in every figure.
COLORS = {
    "random": "#8a8985",
    "greedy": "#3d3c3a",
    "sarsa_timer": "#2a78d6",
    "sarsa_no_timer": "#eb6834",
    "sarsa_fixed_eps": "#1baf7a",
}

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e4e3df",
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
})


def read_csv(name):
    with open(os.path.join(RESULTS, name), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_summary():
    return {row["agent"]: row for row in read_csv("summary.csv")}


def load_curves():
    """Returns {agent: (episodes, array of shape (n_seeds, n_points))}."""
    rows = read_csv("eval_curve.csv")
    curves = {}
    for agent in dict.fromkeys(r["agent"] for r in rows):
        seeds = sorted({int(r["seed"]) for r in rows if r["agent"] == agent})
        per_seed = []
        for seed in seeds:
            pts = [r for r in rows if r["agent"] == agent and int(r["seed"]) == seed]
            per_seed.append([float(r["mean_eval_return"]) for r in pts])
            episodes = [int(r["episodes_trained"]) for r in pts]
        curves[agent] = (np.array(episodes), np.array(per_seed))
    return curves


def load_training_returns():
    """Returns {agent: array of shape (n_seeds, n_episodes)} of raw training returns."""
    with open(os.path.join(RESULTS, "training_returns.csv"), newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        data = np.array([[float(v) for v in row] for row in reader])
    out = {}
    for col, name in enumerate(header[1:], start=1):
        agent = name.rsplit("_seed", 1)[0]
        out.setdefault(agent, []).append(data[:, col])
    return {agent: np.array(cols) for agent, cols in out.items()}


def moving_average(x, window):
    return np.convolve(x, np.ones(window) / window, mode="valid")


def plot_mean_std(ax, x, values, agent, label):
    """Mean line with a shaded +-1 std band (std over seeds)."""
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros_like(mean)
    ax.plot(x, mean, color=COLORS[agent], linewidth=2, label=label)
    ax.fill_between(x, mean - std, mean + std, color=COLORS[agent], alpha=0.18, linewidth=0)


def baseline_lines(ax, summary):
    for agent, label in [("greedy", "Greedy (fixed rule)"), ("random", "Random")]:
        ax.axhline(float(summary[agent]["mean_return"]), color=COLORS[agent],
                   linestyle="--", linewidth=1.5, label=f"{label} (eval mean)")


# ---------------------------------------------------------------------------
def fig0_transition_example():
    row = read_csv("transition_example.csv")[0]
    state = tuple(int(v) for v in row["state"].split())
    next_state = tuple(int(v) for v in row["next_state"].split())
    action = ACTION_NAMES[int(row["action"])]
    reward = float(row["reward"])

    fig, axes = plt.subplots(1, 2, figsize=(9, 5.6))
    for ax, s, title in [(axes[0], state, "State S"), (axes[1], next_state, "Next state S'")]:
        ax_, ay_, px, py, carrying, timer = s
        ax.set_xlim(-0.5, N - 0.5)
        ax.set_ylim(-0.5, N - 0.5)
        ax.set_aspect("equal")
        ax.set_xticks(range(N))
        ax.set_yticks(range(N))
        ax.grid(False)
        for k in range(N + 1):
            ax.axhline(k - 0.5, color="#c3c2b7", linewidth=1)
            ax.axvline(k - 0.5, color="#c3c2b7", linewidth=1)
        ax.add_patch(plt.Rectangle((BASKET[0] - 0.45, BASKET[1] - 0.45), 0.9, 0.9,
                                   color="#eda100", alpha=0.35))
        ax.text(BASKET[0], BASKET[1] - 0.32, "basket", ha="center", fontsize=9)
        if not carrying:
            ax.plot(px, py, "o", color="#e34948", markersize=26)          # apple on the ground
            ax.text(px + 0.3, py + 0.32, f"timer={timer}", fontsize=8)
        ax.plot(ax_, ay_, "o", color="#2a78d6", markersize=14,          # agent
                markeredgecolor="#e34948" if carrying else "white", markeredgewidth=4)
        ax.set_title(f"{title}\n{s}", fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    legend_items = [
        plt.Line2D([], [], marker="o", linestyle="", color="#2a78d6", markersize=10, label="agent"),
        plt.Line2D([], [], marker="o", linestyle="", color="#e34948", markersize=12, label="apple"),
        plt.Line2D([], [], marker="o", linestyle="", color="#2a78d6", markeredgecolor="#e34948",
                   markeredgewidth=3, markersize=10, label="agent carrying apple"),
    ]
    fig.legend(handles=legend_items, loc="upper center", ncol=3, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 0.89))
    fig.suptitle(f"One transition: action = {action}, reward R = {reward:+.2f}\n"
                 "state = (agent_x, agent_y, apple_x, apple_y, carrying, timer)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(os.path.join(RESULTS, "fig0_transition_example.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def fig1_training_curve(summary, curves, train_returns):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # (a) greedy evaluation return during training (E1, E3)
    ax = axes[0]
    for agent, label in [("sarsa_timer", "SARSA (timer)"), ("sarsa_no_timer", "SARSA (no timer)")]:
        if agent in curves:
            x, values = curves[agent]
            plot_mean_std(ax, x, values, agent, f"{label}, mean ± 1 std")
    baseline_lines(ax, summary)
    n_seeds = curves["sarsa_timer"][1].shape[0]
    ax.set_title(f"(a) Greedy evaluation return during training ({n_seeds} seeds)")
    ax.set_xlabel("Training episodes")
    ax.set_ylabel("Mean evaluation return (100 episodes, ε = 0)")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.07), fontsize=9)

    # (b) raw training return of the behaviour (epsilon-greedy) policy, smoothed
    ax = axes[1]
    for agent, label in [("sarsa_timer", "SARSA (timer)"), ("sarsa_no_timer", "SARSA (no timer)")]:
        if agent in train_returns:
            values = train_returns[agent]
            window = max(1, min(200, values.shape[1] // 10))
            smoothed = np.array([moving_average(v, window) for v in values])
            x = np.arange(window, values.shape[1] + 1)
            plot_mean_std(ax, x, smoothed, agent, f"{label}, mean ± 1 std")
    ax.set_title(f"(b) Training return, ε-greedy ({window}-episode moving avg.)")
    ax.set_xlabel("Training episodes")
    ax.set_ylabel("Return per training episode")
    ax.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig1_training_curve.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
def fig2_final_comparison(summary):
    agents = [a for a in ["random", "greedy", "sarsa_timer", "sarsa_no_timer"] if a in summary]
    labels = ["Random", "Greedy\n(fixed rule)", "SARSA\n(timer)", "SARSA\n(no timer)"][:len(agents)]
    colors = [COLORS[a] for a in agents]
    n_seeds = summary[agents[0]]["n_seeds"]
    n_eval = summary[agents[0]]["eval_episodes_per_seed"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    panels = [("mean_return", "std_return", "Mean evaluation return per episode",
               "(a) Final evaluation return"),
              ("mean_delivered", "std_delivered", "Apples delivered per episode",
               "(b) Apples delivered")]
    for ax, (mean_key, std_key, ylabel, title) in zip(axes, panels):
        means = [float(summary[a][mean_key]) for a in agents]
        stds = [float(summary[a][std_key]) for a in agents]
        bars = ax.bar(range(len(agents)), means, yerr=stds, color=colors, width=0.6,
                      capsize=5, edgecolor="white", linewidth=2)
        for bar, m, sd in zip(bars, means, stds):
            # write the value just above the error bar (above zero for negative bars)
            ax.annotate(f"{m:.2f}", (bar.get_x() + bar.get_width() / 2, max(m + sd, 0)),
                        xytext=(0, 6), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)
        ax.axhline(0, color="#52514e", linewidth=1)
        ax.set_xticks(range(len(agents)))
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    fig.suptitle(f"Final comparison: greedy evaluation, mean ± std over {n_seeds} seeds "
                 f"({n_eval} episodes per seed)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig2_final_comparison.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
def fig3_epsilon_ablation(summary, curves):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for agent, label in [("sarsa_timer", "Decaying ε (1.0 → 0.05)"),
                         ("sarsa_fixed_eps", "Fixed ε = 0.1")]:
        x, values = curves[agent]
        plot_mean_std(ax, x, values, agent, f"{label}, mean ± 1 std")
    baseline_lines(ax, summary)
    ax.set_title("E4: exploration schedule (SARSA with timer, 5 seeds)")
    ax.set_xlabel("Training episodes")
    ax.set_ylabel("Mean evaluation return (100 episodes, ε = 0)")
    ax.legend(loc="center right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig3_epsilon_ablation.png"), dpi=150)
    plt.close(fig)


def write_curve_summary(curves, train_returns):
    """Small tables used in the report text:
    curve_summary.csv    mean / std / min over seeds of the greedy evaluation return
                         at every checkpoint
    training_windows.csv mean training return (over seeds and episodes) in fixed windows
    """
    with open(os.path.join(RESULTS, "curve_summary.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["agent", "episodes_trained", "mean", "std", "min_over_seeds",
                         "seeds_below_100"])
        for agent, (x, values) in curves.items():
            for i, ep in enumerate(x):
                col = values[:, i]
                std = col.std(ddof=1) if len(col) > 1 else 0.0
                writer.writerow([agent, ep, round(col.mean(), 2), round(std, 2),
                                 round(col.min(), 2), int((col < 100).sum())])

    with open(os.path.join(RESULTS, "training_windows.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["agent", "first_episode", "last_episode", "mean_training_return"])
        for agent, values in train_returns.items():
            n = values.shape[1]
            window = max(1, n // 10)                 # 10 windows (2000 episodes in the full run)
            for start in range(0, n, window):
                writer.writerow([agent, start + 1, min(start + window, n),
                                 round(values[:, start:start + window].mean(), 2)])


def make_all_figures():
    summary = load_summary()
    curves = load_curves()
    train_returns = load_training_returns()
    write_curve_summary(curves, train_returns)
    fig0_transition_example()
    fig1_training_curve(summary, curves, train_returns)
    fig2_final_comparison(summary)
    if "sarsa_fixed_eps" in curves:
        fig3_epsilon_ablation(summary, curves)


if __name__ == "__main__":
    make_all_figures()
    print("figures written to", RESULTS)
