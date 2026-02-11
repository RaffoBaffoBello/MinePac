import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

LOG_PATH = Path(__file__).resolve().parent / "az_log.jsonl"
OUT_PATH = Path(__file__).resolve().parent / "az_progress.png"


def load_rows(path):
    if not path.exists():
        raise FileNotFoundError(f"Log not found: {path}")
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def rolling_avg(values, window):
    if window <= 1:
        return values
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = values[start : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def main():
    rows = load_rows(LOG_PATH)
    if not rows:
        print("No data yet in az_log.jsonl")
        return

    scores = [r["score"] for r in rows]
    steps = [r["steps"] for r in rows]
    episodes = list(range(1, len(rows) + 1))

    score_avg = rolling_avg(scores, window=10)
    step_avg = rolling_avg(steps, window=10)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_title("AlphaZero Self-Play Progress")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Score", color="tab:blue")
    ax1.plot(episodes, scores, color="tab:blue", alpha=0.3, label="Score")
    ax1.plot(episodes, score_avg, color="tab:blue", label="Score (avg 10)")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Steps", color="tab:green")
    ax2.plot(episodes, steps, color="tab:green", alpha=0.3, label="Steps")
    ax2.plot(episodes, step_avg, color="tab:green", label="Steps (avg 10)")
    ax2.tick_params(axis="y", labelcolor="tab:green")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Saved plot to {OUT_PATH}")

    print(
        f"Episodes: {len(scores)} | score avg: {statistics.mean(scores):.1f} | score max: {max(scores)} | steps avg: {statistics.mean(steps):.1f}"
    )


if __name__ == "__main__":
    main()
