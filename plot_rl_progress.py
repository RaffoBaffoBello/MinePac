import json
import os

import matplotlib.pyplot as plt


LOG_PATH = os.path.join(os.path.dirname(__file__), "rl_log.jsonl")
OUT_PATH = os.path.join(os.path.dirname(__file__), "rl_progress.png")


def main():
    if not os.path.exists(LOG_PATH):
        raise SystemExit("rl_log.jsonl not found.")

    steps = []
    avg_rewards = []
    eps_vals = []
    scores = []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            steps.append(int(data.get("steps", 0)))
            avg_rewards.append(float(data.get("avg_reward_200", 0.0)))
            eps_vals.append(float(data.get("eps", 0.0)))
            scores.append(int(data.get("last_score", 0)))

    if not steps:
        raise SystemExit("No log entries found.")

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(steps, avg_rewards, label="Avg Reward (200)", color="tab:blue")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Avg Reward", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(steps, scores, label="Score", color="tab:green")
    ax2.plot(steps, eps_vals, label="Epsilon", color="tab:red", alpha=0.6)
    ax2.set_ylabel("Score / Epsilon")
    fig.tight_layout()
    plt.savefig(OUT_PATH, dpi=160)
    print(f"Saved plot to {OUT_PATH}")


if __name__ == "__main__":
    main()
