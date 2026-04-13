"""
Plotly visualisations
    metrics  (default)
        Line chart with dual dropdowns — one for LCP name, one for metric.

    diff
        Line chart of ROC-AUC and PR-AUC differences between models.
"""

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

# All LCP dataset names
_DATA_NAMES = [
    "Repeats1", "Repeats2", "Repeats3",
    "Order1",   "Order2",   "Order3",
    "Timing1",  "Timing2",  "Timing3", "Timing4",
    "Period1",  "Period2",  "Period3", "Period4",
]

_METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "av_prec", "brier", "prc_auc"]

_DIFF_METRICS = ["roc_auc_diff", "prc_auc_diff"]

def plot_metrics(
    csv_file: str | Path,
    model_label: str = "Model",
) -> go.Figure:
    data = pd.read_csv(csv_file).sort_values("data_size")
    fig  = go.Figure()

    for category in _DATA_NAMES:
        for metric in _METRICS:
            if metric not in data.columns:
                continue
            subset = data[data["data_name"] == category]
            fig.add_trace(go.Scatter(
                x=subset["data_size"], y=subset[metric],
                mode="lines+markers",
                name=f"{category} — {metric}",
                line=dict(width=2),
                visible=False,
            ))

    # Dropdown: LCP name
    lcp_buttons = [
        dict(
            label=cat, method="update",
            args=[
                {"visible": [cat in t.name for t in fig.data]},
                {"title": f"LCP Performance vs Training Set Size — {cat} ({model_label})"},
            ],
        )
        for cat in _DATA_NAMES
    ]

    # Dropdown: metric
    metric_buttons = [
        dict(
            label=met, method="update",
            args=[
                {"visible": [t.name.endswith(f"— {met}") for t in fig.data]},
                {"title": f"{met.upper()} vs Training Set Size ({model_label})"},
            ],
        )
        for met in _METRICS
    ]

    fig.update_layout(
        updatemenus=[
            dict(buttons=lcp_buttons,    direction="down", showactive=True,
                 x=1.00, xanchor="left", y=1.15, yanchor="top"),
            dict(buttons=metric_buttons, direction="down", showactive=True,
                 x=1.18, xanchor="left", y=1.15, yanchor="top"),
        ],
        title=f"LCP Performance vs Training Set Size ({model_label})",
        xaxis_title="Training Set Size",
        yaxis_title="Metric Value",
        template="plotly_dark",
        legend_title="LCP — Metric",
        hovermode="x unified",
        font=dict(size=14),
    )
    return fig


def plot_diff(
    csv_file: str | Path,
    model_label: str = "Model",
) -> go.Figure:
    data = pd.read_csv(csv_file).sort_values("data_size")
    fig  = go.Figure()

    for category in _DATA_NAMES:
        for metric in _DIFF_METRICS:
            if metric not in data.columns:
                continue
            subset = data[data["data_name"] == category]
            fig.add_trace(go.Scatter(
                x=subset["data_size"], y=subset[metric],
                mode="lines+markers",
                name=f"{category} — {metric}",
                line=dict(width=2),
                visible=False,
            ))

    lcp_buttons = [
        dict(
            label=cat, method="update",
            args=[
                {"visible": [cat in t.name for t in fig.data]},
                {"title": f"Performance Difference vs Training Set Size — {cat}"},
            ],
        )
        for cat in _DATA_NAMES
    ]

    metric_buttons = [
        dict(
            label=met, method="update",
            args=[
                {"visible": [t.name.endswith(f"— {met}") for t in fig.data]},
                {"title": f"{met} vs Training Set Size ({model_label})"},
            ],
        )
        for met in _DIFF_METRICS
    ]

    fig.update_layout(
        updatemenus=[
            dict(buttons=lcp_buttons,    direction="down", showactive=True,
                 x=1.00, xanchor="left", y=1.15, yanchor="top"),
            dict(buttons=metric_buttons, direction="down", showactive=True,
                 x=1.18, xanchor="left", y=1.15, yanchor="top"),
        ],
        title=f"Capacity Degradation Analysis ({model_label})",
        xaxis_title="Training Set Size",
        yaxis_title="Metric Difference",
        template="plotly_dark",
        legend_title="LCP — Metric",
        hovermode="x unified",
        font=dict(size=14),
    )
    return fig

def _parse_args():
    parser = argparse.ArgumentParser(description="Interactive LCP performance visualiser")
    parser.add_argument("--csv",   required=True, help="Path to the results CSV file")
    parser.add_argument("--mode",  default="metrics", choices=["metrics", "diff"],
                        help="Chart type: 'metrics' (default) or 'diff'")
    parser.add_argument("--model", default="Model",
                        help="Model label shown in chart titles (e.g. ResNet)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.mode == "diff":
        fig = plot_diff(args.csv, model_label=args.model)
    else:
        fig = plot_metrics(args.csv, model_label=args.model)
    fig.show()
