"""Export a loss figure for the retained 1..500 -> resumed 501..2500 path.

The append-only log includes discarded startup runs and an initial branch up to
670. The actual checkpoint lineage resumes at 500, so that branch's >500 rows
must not be connected to the resumed branch.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/action-refiner-bs768/metrics.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/action-refiner-bs768/plots"))
    args = parser.parse_args()
    segments = []
    for line in args.input.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not segments or row["step"] <= segments[-1][-1]["step"]:
            segments.append([])
        segments[-1].append(row)
    if len(segments) != 4 or segments[2][0]["step"] != 1 or segments[3][0]["step"] != 501:
        raise ValueError("Log lineage changed; review segments instead of silently joining runs")
    rows = [dict(row, source_segment=2) for row in segments[2] if row["step"] <= 500]
    rows += [dict(row, source_segment=3) for row in segments[3]]
    x = np.array([row["step"] for row in rows])
    y = np.array([row["train_flow_loss"] for row in rows])
    if not np.all(np.diff(x) > 0) or not np.all(np.isfinite(y)):
        raise ValueError("Selected steps must be increasing with finite loss")
    # Smooth only reviewed log points, not invented per-update observations.
    smoothed = np.array([np.mean(y[max(0, index-10):index+1]) if index >= 10 else np.nan
                         for index in range(len(y))])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_UTC")
    stem = args.output_dir / f"action_refiner_training_loss_{stamp}"
    fields = ["step", "train_flow_loss", "grad_norm", "lr", "elapsed_seconds", "source_segment"]
    with stem.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#30343B", "text.color": "#30343B",
                         "xtick.color": "#505762", "ytick.color": "#505762"})
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.4), gridspec_kw={"height_ratios": [1.1, 1]})
    fig.subplots_adjust(left=0.10, right=0.965, top=0.86, bottom=0.17, hspace=0.48)
    fig.suptitle("Action Flow Refiner — Training Loss", x=0.10, y=0.97, ha="left", fontsize=20, weight="bold")
    fig.text(0.10, 0.917, "Frozen Lingbot-VLA-V2  |  global batch 768  |  masked velocity L1  |  1–2500 updates",
             fontsize=11, color="#59616B")
    for ax, start in zip(axes, [1, 500]):
        keep = x >= start
        ax.plot(x[keep], y[keep], color="#7EA9D0", alpha=0.75, linewidth=0.9, label="Logged batch loss")
        ax.plot(x[keep], smoothed[keep], color="#174C78", linewidth=2.0,
                label="Trailing mean · 11 logged points")
        ax.set_xlim(start, int(x[-1]))
        ax.set_ylabel("Flow L1 loss")
        ax.set_xlabel("Optimizer update")
        ax.grid(axis="y", color="#E3E7EC", linewidth=0.7)
        ax.xaxis.set_major_locator(MultipleLocator(500))
        for step in [500, 1500, 2500]:
            ax.axvline(step, color="#88919B", linestyle=(0, (3, 4)), linewidth=0.8, zorder=0)
        if start == 1:
            ax.axvspan(1, 100, color="#E9D6A5", alpha=0.35, zorder=0)
            ax.set_title("Full training path · warmup shaded through update 100", loc="left", fontsize=11, pad=10)
            ax.legend(loc="upper right", frameon=False, ncol=2, fontsize=10)
        else:
            ax.set_title("Detail after update 500 · focused vertical scale", loc="left", fontsize=11, pad=10)
            ymin, ymax = y[keep].min(), y[keep].max()
            margin = (ymax - ymin) * 0.17
            ax.set_ylim(ymin-margin, ymax+margin)
            for step, offset in [(500, (12, 9)), (1500, (0, 18)), (2500, (-14, -23))]:
                index = np.where(x == step)[0][0]
                ax.scatter([step], [y[index]], s=24, color="#174C78", zorder=4, clip_on=False)
                ax.annotate(f"{step}: {y[index]:.5f}", (step, y[index]), xytext=offset,
                            textcoords="offset points", fontsize=9,
                            ha="left" if step == 500 else "right" if step == 2500 else "center")
    fig.text(0.10, 0.085,
             "Lineage: initial run updates 1–500 → restored checkpoint → updates 501–2500.\n"
             "Excluded: two early test runs and the superseded initial-run updates 510–670.\n"
             "Loss is logged every 10 updates plus first/resume updates; it is not rollout success or final action error.",
             fontsize=9, color="#59616B", va="center", linespacing=1.5)
    png = stem.with_suffix(".png")
    fig.savefig(png, dpi=260, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    receipt = {"source": str(args.input.resolve()), "points": len(rows),
               "selected_lineage": ["segment 2: updates <=500", "segment 3: updates >=501"],
               "segments": [{"segment": i, "first_step": seg[0]["step"], "last_step": seg[-1]["step"],
                             "points": len(seg)} for i, seg in enumerate(segments)],
               "anchors": {str(step): float(y[np.where(x == step)[0][0]]) for step in [1, 500, 1500, 2500]},
               "smoothing": "Trailing arithmetic mean of 11 recorded points; no interpolation of missing updates"}
    stem.with_suffix(".json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps({"png": str(png.resolve()), "csv": str(stem.with_suffix('.csv').resolve()),
                      "receipt": receipt}, indent=2))


if __name__ == "__main__":
    main()
