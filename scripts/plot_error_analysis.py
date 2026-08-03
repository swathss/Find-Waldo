from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main() -> None:
    parser = argparse.ArgumentParser(description="Create poster-ready plots from evaluation CSV files.")
    parser.add_argument("--metrics", type=Path, required=True, nargs="+", help="method=path.csv")
    parser.add_argument("--errors", type=Path, required=True, nargs="+", help="method=path.csv")
    parser.add_argument("--output", type=Path, default=Path("artifacts/figures"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")

    metric_frames = []
    for specification in args.metrics:
        method, path = str(specification).split("=", 1)
        frame = pd.read_csv(path)
        frame["method"] = method
        metric_frames.append(frame)
    metrics = pd.concat(metric_frames, ignore_index=True)
    long_metrics = metrics.melt(
        id_vars=["method", "class_name"], value_vars=["precision", "recall", "f1", "ap50"],
        var_name="metric", value_name="value"
    )
    figure, axis = plt.subplots(figsize=(12, 6))
    sns.barplot(data=long_metrics, x="class_name", y="value", hue="method", errorbar=None, ax=axis)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Class")
    axis.set_ylabel("Score")
    axis.set_title("Detection performance by class")
    figure.tight_layout()
    figure.savefig(args.output / "performance_by_class.png", dpi=220)
    plt.close(figure)

    error_frames = []
    for specification in args.errors:
        method, path = str(specification).split("=", 1)
        frame = pd.read_csv(path)
        frame["method"] = method
        error_frames.append(frame)
    errors = pd.concat(error_frames, ignore_index=True)
    counts = errors.groupby(["method", "outcome"], as_index=False).size()
    figure, axis = plt.subplots(figsize=(9, 5))
    sns.barplot(data=counts, x="outcome", y="size", hue="method", ax=axis)
    axis.set_xlabel("Outcome")
    axis.set_ylabel("Count")
    axis.set_title("True positives, false positives and false negatives")
    figure.tight_layout()
    figure.savefig(args.output / "error_counts.png", dpi=220)
    plt.close(figure)


if __name__ == "__main__":
    main()

