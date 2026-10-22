"""Render the résumé chart from the saved query-level benchmark artifact."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
report = json.loads((root / "Prototype/evaluation_results/msrvtt_results.json").read_text())
baseline = report["first_frame_baseline_held_out"]["recall_at_1"] * 100
sampled = report["held_out"]["recall_at_1"] * 100
count = report["held_out"]["queries"]
plt.rcParams["svg.fonttype"] = "none"
fig, ax = plt.subplots(figsize=(7, 3.8))
bars = ax.barh(["First frame", "Sampled frames"], [baseline, sampled], color=["#94a3b8", "#2563eb"])
ax.set_xlim(0, 100)
ax.set_xlabel("Held-out Recall@1 (%)")
ax.set_title(f"MomentFlow visual retrieval · {count} queries, 50 candidate videos")
for bar, value in zip(bars, [baseline, sampled]):
    ax.text(value + 1, bar.get_y() + bar.get_height() / 2, f"{value:.0f}%", va="center")
ax.spines[["top", "right"]].set_visible(False)
fig.text(0.5, 0.01, "Fixed MSR-VTT subset; local visual-only evaluation, not full benchmark accuracy", ha="center", fontsize=8)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(root / "docs/recall_chart.svg")
