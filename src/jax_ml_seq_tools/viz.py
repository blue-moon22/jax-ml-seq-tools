"""Plotting and metric-dataframe utilities."""

from __future__ import annotations

from collections import OrderedDict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

NAMED_COLORS = OrderedDict(
    [
        ("red", "#e41a1c"),
        ("blue", "#377eb8"),
        ("green", "#4daf4a"),
        ("purple", "#984ea3"),
        ("orange", "#ff7f00"),
        ("yellow", "#ffff33"),
        ("brown", "#a65628"),
        ("pink", "#f781bf"),
        ("gray", "#999999"),
    ]
)

DEFAULT_SPLIT_COLORS = {
    "train": NAMED_COLORS["blue"],
    "valid": NAMED_COLORS["green"],
    "test": NAMED_COLORS["orange"],
}


def plot_binding_site(panels, highlight: tuple[int, int] | None = None):
    """Plot a line plot and heatmap of contribution scores in a DNA sequence."""
    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(7, 3),
        gridspec_kw={"height_ratios": [1, 3]},
        sharex=True,
        constrained_layout=True,
    )

    ax1, ax2 = axes

    ax1.plot(panels["line"]["values"], c="black")
    ax1.set_ylabel(panels["line"]["label"])
    for spine in ax1.spines.values():
        spine.set_visible(False)
    ax1.set_xticks([])

    heatmap = sns.heatmap(
        panels["tiles"]["values"].T,
        ax=ax2,
        center=0,
        cbar=False,
        cmap="viridis",
        yticklabels=["A", "C", "G", "T"],
    )
    ax2.set_xlabel("Position in DNA sequence")
    ax2.set_ylabel("DNA Base")

    cbar = fig.colorbar(
        heatmap.collections[0],
        ax=axes,
        orientation="vertical",
        fraction=0.02,
        pad=0.02,
    )
    cbar.set_label(panels["tiles"]["label"])

    if highlight:
        start, end = highlight
        rect = plt.Rectangle(
            xy=(start, 0),
            width=end - start,
            height=panels["tiles"]["values"].shape[1],
            linewidth=3,
            edgecolor="black",
            facecolor="none",
        )
        ax2.add_patch(rect)

    return fig


def to_df(exported_metrics: dict[str, dict[str, list[dict]]]) -> pd.DataFrame:
    """Convert exported MetricsLogger history into a long-form DataFrame."""
    rows = []
    for split, metrics in exported_metrics.items():
        for metric, records in metrics.items():
            for record in records:
                rows.append(
                    {
                        "split": split,
                        "metric": metric,
                        "round": record["round"],
                        "mean": record["mean"],
                        "std": record["std"],
                        "unit": record["unit"],
                    }
                )
    return pd.DataFrame(rows)
